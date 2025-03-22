import argparse
import datetime
import os
import random
import time
import numpy as np
import torch

from pysgg.engine.trainer import reduce_loss_dict
from pysgg.modeling.detector import build_detection_model
from pysgg.solver import make_optimizer
from pysgg.utils.checkpoint import DetectronCheckpointer
from pysgg.utils.checkpoint import clip_grad_norm
from pysgg.utils import visualize_graph as vis_graph
from pysgg.utils.comm import synchronize, get_rank
from pysgg.utils.logger import setup_logger, debug_print, TFBoardHandler_LEVEL
from pysgg.utils.metric_logger import MetricLogger
from pysgg.utils.miscellaneous import mkdir
from pysgg.solver.lr_scheduler import WarmupMultiStepLR

from util.dataset import make_data_loader
from util.model import build_detection_model
from util.checkpoint import DetectronCheckpointer
from util.inference import run_test,run_val

SEED = 666

torch.cuda.manual_seed(SEED)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(SEED)  # 为所有GPU设置随机种子
torch.manual_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

torch.backends.cudnn.enabled = True  # 默认值
torch.backends.cudnn.benchmark = True  # 默认为False
torch.backends.cudnn.deterministic = True  # 默认为False;benchmark为True时,y要排除随机性必须为True


torch.autograd.set_detect_anomaly(True)

SHOW_COMP_GRAPH = False

def make_optimizer(args, model, logger, slow_heads=None, except_weight_decay=None, slow_ratio=5.0, rl_factor=1.0):
    params = []
    for key, value in model.named_parameters():
        if not value.requires_grad:
            continue
        lr = args.base_lr
        weight_decay = args.weight_decay
        if "bias" in key:
            lr = args.base_lr * args.bias_factor
            weight_decay = args.weight_decay_bias

        if except_weight_decay is not None:
            for item in except_weight_decay:
                if item in key:
                    weight_decay = 0.0
                    logger.info("NO WEIGHT DECAY: {}.".format(key))

        if slow_heads is not None:
            for item in slow_heads:
                if item in key:
                    logger.info("SLOW HEADS: {} is slow down by ratio of {}.".format(key, str(slow_ratio)))
                    lr = lr / slow_ratio
                    break
        params += [{"params": [value], "lr": lr * rl_factor, "weight_decay": weight_decay}]

    optimizer = torch.optim.SGD(params, lr=args.base_lr, momentum=args.momentum)
    return optimizer

def show_params_status(model):
    """
    Prints parameters of a model
    """
    st = {}
    strings = []
    total_params = 0
    trainable_params = 0
    for p_name, p in model.named_parameters():

        if not ("bias" in p_name.split(".")[-1] or "bn" in p_name.split(".")[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
        if p.requires_grad:
            trainable_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in st.items():
        strings.append(
            "{:<80s}: {:<16s}({:8d}) ({})".format(
                p_name, "[{}]".format(",".join(size)), prod, "grad" if p_req_grad else "    "
            )
        )
    strings = "\n".join(strings)
    return (
        f"\n{strings}\n ----- \n \n"
        f"      trainable parameters:  {trainable_params/ 1e6:.3f}/{total_params / 1e6:.3f} M \n "
    )

def fix_eval_modules(eval_modules):
    for module in eval_modules:
        if module is None:
            continue

        for _, param in module.named_parameters():
            param.requires_grad = False
        # DO NOT use module.eval(), otherwise the module will be in the test mode, i.e., all self.training condition is set to False

def set_train_modules(modules):
    for module in modules:
        for _, param in module.named_parameters():
            param.requires_grad = True

def train(args,local_rank,distributed,logger):
    global SHOW_COMP_GRAPH
    #初始化数据集和数据加载器
    debug_print(logger, "Start initializing dataset & dataloader")
    #字典 arguments，用于存储训练过程中的参数。
    arguments = {}
    arguments["iteration"] = 0
    #设置输出目录，用于保存训练结果（如模型权重、日志文件等）。
    output_dir = args.output_dir
    #创建训练集和验证集的数据加载器
    train_loader = make_data_loader(args,mode='train',  
        is_distributed=distributed,
        start_iter=arguments["iteration"],)
    val_loader  = make_data_loader(args,mode = 'val', 
            is_distributed=distributed,
        start_iter=arguments["iteration"],)
    logger.info(f'the nums of train loader is {len(train_loader)}')
    logger.info(f'the nums of val loader is {len(val_loader)}')

    debug_print(logger, "end dataloader")
    debug_print(logger, "prepare training")
    #模型构造
    model = build_detection_model(args)
    model.train()
    debug_print(logger, "end model construction")
    logger.info(str(model))
    eval_modules = (
        model.rpn,
        model.backbone,
        model.roi_heads.box,
    )
    train_modules = ()
    rel_pn_module_ref = []
    fix_eval_modules(eval_modules)
    set_train_modules(train_modules)
    if model.roi_heads.relation.rel_pn is not None:
        rel_on_module = (model.roi_heads.relation.rel_pn,)
    else:
        rel_on_module = None
    logger.info("trainable models:")
    logger.info(show_params_status(model))
    slow_heads = []
    except_weight_decay = []
    # load pretrain layers to new layers
    load_mapping = {
        "roi_heads.relation.box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.rel_pair_box_feature_extractor": "roi_heads.box.feature_extractor",
        "roi_heads.relation.union_feature_extractor.feature_extractor": "roi_heads.box.feature_extractor",
    }

    print("load model to GPU")
    device = torch.device(args.device)
    model.to(device)
    #创建优化器和学习率调度器
    optimizer = make_optimizer(
        args,
        model,
        logger,
        slow_heads=slow_heads,
        slow_ratio=2.5,
        rl_factor=4.0,
        # rl_factor=1.0,
        except_weight_decay=except_weight_decay,
    )
    scheduler = WarmupMultiStepLR(
            optimizer,
            [40000],
            0.5,
            warmup_factor=0.1,
            warmup_iters=500,
            warmup_method='linear',
        )
    debug_print(logger, "end optimizer and shcedule")
    scaler = torch.cuda.amp.GradScaler()

    save_to_disk = get_rank() == 0
    checkpointer = DetectronCheckpointer(
        args, model, optimizer, scheduler, output_dir, save_to_disk, custom_scheduler=True
    )
    checkpointer.load(
        args.pretrained_detection_ckpt, with_optim=False, load_mapping=load_mapping
    )
    checkpoint_period = args.checkpoint_period

     # preserve a reference for logging
    rel_model_ref = model.roi_heads.relation

    if distributed:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            # this should be removed if we update BatchNorm stats
            broadcast_buffers=False,
            find_unused_parameters=True,
        )
    debug_print(logger, "end distributed")
    # logger.info("Validate before training")
    # run_val(args, model, val_loader, distributed, logger)
    # pre_clser_pretrain_on = False
    if distributed:
        m2opt = model.module
    else:
        m2opt = model
    m2opt.roi_heads.relation.predictor.start_preclser_relpn_pretrain()
    logger.info("Start preclser_relpn_pretrain")
    pre_clser_pretrain_on = True

    STOP_ITER = (2000)

    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    max_iter = len(train_loader)
    start_iter = arguments["iteration"]
    start_training_time = time.time()
    end = time.time()
    model.train()

    print_first_grad = True
    for iteration, (images, targets, _) in enumerate(train_loader, start_iter):
        if any(len(target) < 1 for target in targets):
            logger.error(
                f"Iteration={iteration + 1} || Image Ids used for training {_} || targets Length={[len(target) for target in targets]}"
            )
        data_time = time.time() - end
        iteration = iteration + 1
        arguments["iteration"] = iteration

        model.train()

        fix_eval_modules(eval_modules)
        optimizer.zero_grad()
        scheduler.step()
        with torch.cuda.amp.autocast():
            images = images.to(device)
            targets = [target.to(device) for target in targets]

            loss_dict = model(images, targets, logger=logger)

            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_loss_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        meters.update(loss=losses_reduced, **loss_dict_reduced)
        
        # Note: If mixed precision is not used, this ends up doing nothing
        # Otherwise apply loss scaling for mixed-precision recipe
        # try:
        # with amp.scale_loss(losses, optimizer) as scaled_losses:
        #     scaled_losses.backward()
        scaler.scale(losses).backward()
        # 
        # 

        if not SHOW_COMP_GRAPH and get_rank() == 0:
            try:
                g = vis_graph.visual_computation_graph(
                    losses, model.named_parameters(), args.output_dir, "total_loss-graph"
                )
                g.render()
                for name, ls in loss_dict_reduced.items():
                    g = vis_graph.visual_computation_graph(
                        losses, model.named_parameters(), args.output_dir, f"{name}-graph"
                    )
                    g.render()
            except:
                logger.info("print computational graph failed")

            SHOW_COMP_GRAPH = True

        # add clip_grad_norm from MOTIFS, tracking gradient, used for debug
        verbose = (
            iteration % 4000
        ) == 0 or print_first_grad  # print grad or not
        print_first_grad = False
        clip_grad_norm(
            [(n, p) for n, p in model.named_parameters() if p.requires_grad],
            max_norm=5.0,
            logger=logger,
            verbose=verbose,
            clip=True,
        )

        scaler.step(optimizer)

        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)

        eta_seconds = meters.time.global_avg * (max_iter - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        elapsed_time = str(datetime.timedelta(seconds=int(end - start_training_time)))

        if (
            iteration
            in [3000,]
            and rel_on_module is not None
        ):
            logger.info("fix the rel pn module")
            fix_eval_modules(rel_pn_module_ref)

        if pre_clser_pretrain_on:
            if iteration == STOP_ITER:
                logger.info("pre clser pretraining ended.")
                m2opt.roi_heads.relation.predictor.end_preclser_relpn_pretrain()
                pre_clser_pretrain_on = False

        if iteration % 30 == 0:
            logger.log(TFBoardHandler_LEVEL, (meters.meters, iteration))

            logger.log(
                TFBoardHandler_LEVEL,
                ({"curr_lr": float(optimizer.param_groups[0]["lr"])}, iteration),
            )
            # save_buffer(output_dir)

        if iteration % 100 == 0 or iteration == max_iter:
            logger.info(
                meters.delimiter.join(
                    [
                        "\ninstance name: {instance_name}\n" "elapsed time: {elapsed_time}\n",
                        "eta: {eta}\n",
                        "iter: {iter}/{max_iter}\n",
                        "{meters}",
                        "lr: {lr:.6f}\n",
                        "max mem: {memory:.0f}\n",
                    ]
                ).format(
                    instance_name=args.output_dir[len("checkpoints/") :],
                    eta=eta_string,
                    elapsed_time=elapsed_time,
                    iter=iteration,
                    meters=str(meters),
                    lr=optimizer.param_groups[0]["lr"],
                    max_iter=max_iter,
                    memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                )
            )
            if pre_clser_pretrain_on:
                logger.info("relness module pretraining..")

        if iteration % checkpoint_period == 0:
            checkpointer.save("model_{:07d}".format(iteration), **arguments)
        if iteration == max_iter:
            checkpointer.save("model_final", **arguments)

        val_result_value = None  # used for scheduler updating
        if iteration % 1000 == 0:
            logger.info("Start validating")
            val_result = run_val(args, model, val_loader, distributed, logger)
            val_result_value = val_result[1]
            if get_rank() == 0:
                for each_ds_eval in val_result[0]:
                    for each_evalator_res in each_ds_eval[1]:
                        logger.log(TFBoardHandler_LEVEL, (each_evalator_res, iteration))
        # scheduler should be called after optimizer.step() in pytorch>=1.1.0
        # https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate
        
        scaler.update()

    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (max_iter)
        )
    )
    return model

def main():
    parser = argparse.ArgumentParser(description="PyTorch Relation Detection Training")
    parser.add_argument(
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)

    parser.add_argument('--output_dir',type=str,default='checkpoints/sgdet-BGNNPredictor')
    parser.add_argument('--dataset_name',type=str,default='VG_stanford')
    parser.add_argument('--dataset_name_test',type=str,default='VG_stanford_test')
    parser.add_argument('--dataset_name_val',type=str,default='VG_stanford_val')
    parser.add_argument('--img_dir',type=str,default='datasets/vg/stanford_spilt/VG_100k_images')
    #parser.add_argument('--glove_dir',type=str,default='datasets/vg/stanford_spilt/glove') 
    #parser.add_argument('--word2vec_dir', type=str, default='datasets/vg/stanford_spilt/word2vec')
    parser.add_argument('--bert_dir', type=str, default='datasets/vg/stanford_spilt/bert')  
    parser.add_argument('--roidb_file',type=str,default='datasets/vg/VG-SGG-with-attri.h5')
    parser.add_argument('--dict_file',type=str,default='datasets/vg/VG-SGG-dicts-with-attri.json')
    parser.add_argument('--image_file',type=str,default='datasets/vg/image_data.json')
    parser.add_argument('--debug',type=bool,default = True)

    parser.add_argument('--train_pre_batch',type=int,default=4)
    parser.add_argument('--test_pre_batch',type=int,default=1)
    parser.add_argument('--max_iter',type=int,default=1000)
    parser.add_argument('--size_divisbility',type=int,default=32)

    parser.add_argument('--base_lr',type=float,default=0.008)
    parser.add_argument('--weight_decay',type=float,default=1.0e-05)
    parser.add_argument('--bias_factor',type=float,default=1.0)
    parser.add_argument('--weight_decay_bias',type=float,default=0.0)
    parser.add_argument('--momentum',type=float,default=0.9)

    parser.add_argument('--pretrained_detection_ckpt',type=str,default='checkpoints/detection/pretrained_faster_rcnn/vg_faster_det.pth')
    parser.add_argument('--checkpoint_period',type=int,default=500)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    #################backbone cfg ############
    parser.add_argument('--res_in_channels',type=int,default=64)
    parser.add_argument('--res_out_channels',type=int,default=256)
    parser.add_argument('--backbone_out_channels',type=int,default=256)
   
    # parser.add_argument('--size_divisbility',type=int,default=32)
    
    #################rpn cfg ############


    #################roi_heads cfg ############
    parser.add_argument('--box_head_num_class',type=int,default=151)

    args = parser.parse_args()
    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://")
        synchronize()

    # mode = "sgcls"
    # now = datetime.datetime.now()
    # time_str = now.strftime("%Y-%m-%d_%H")
    now = datetime.datetime.now()
    time_str = now.strftime("%Y-%m-%d_%H")
    args.output_dir = os.path.join(args.output_dir,time_str) 
    output_dir = args.output_dir
    if output_dir:
        mkdir(output_dir)
    
    logger = setup_logger("pysgg", output_dir, get_rank())
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    model = train(args,args.local_rank, args.distributed,logger)

    run_test(args,model,args.distributed,logger)
    
    print(model)


if __name__=='__main__':
    main()
    