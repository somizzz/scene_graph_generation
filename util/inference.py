import logging
import os
import torch
import pickle
import numpy as np
from matplotlib import pyplot as plt
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from tqdm import tqdm

from pysgg.utils.comm import (
    is_main_process,
    get_world_size,
    synchronize,all_gather
)
from pysgg.utils.timer import Timer,get_time_str
from pysgg.engine.inference import _accumulate_predictions_from_multiple_gpus
from pysgg.data.datasets.evaluation.coco.coco_eval import COCOResults

from pysgg.data.datasets.visual_genome import HEAD, TAIL, BODY
from pysgg.data.datasets.evaluation.vg.vg_eval import save_output,evaluate_relation_of_one_image
from pysgg.utils.miscellaneous import mkdir


from .dataset import make_data_loader
from .sgg_eval import SGRecall, SGNoGraphConstraintRecall, \
    SGZeroShotRecall, SGPairAccuracy, SGMeanRecall, SGStagewiseRecall, SGNGMeanRecall
eval_times = 0
def do_vg_evaluation(
        args,
        dataset,
        predictions,
        output_folder,
        logger,
        iou_types,
):
    # get zeroshot triplet
    zeroshot_triplet = torch.load("pysgg/data/datasets/evaluation/vg/zeroshot_triplet.pytorch",
                                  map_location=torch.device("cpu")).long().numpy()
    attribute_on = False
    num_attributes = 201
    # extract evaluation settings from cfg
    # mode = cfg.TEST.RELATION.EVAL_MODE
    mode = 'sgdet'

    num_rel_category = 51
    multiple_preds = False
    iou_thres = 0.5
    assert mode in {'predcls', 'sgdet', 'sgcls', 'phrdet', 'preddet'}

    groundtruths = []
    for image_id, prediction in enumerate(predictions):
        img_info = dataset.get_img_info(image_id)
        image_width = img_info["width"]
        image_height = img_info["height"]
        # recover original size which is before transform
        predictions[image_id] = prediction.resize((image_width, image_height))

        gt = dataset.get_groundtruth(image_id, evaluation=True)
        groundtruths.append(gt)

    save_output(output_folder, groundtruths, predictions, dataset)
    avg_metrics = 0
    result_str = '\n' + '=' * 100 + '\n'

    result_dict = {}
    result_dict_list_to_log = []

    if "bbox" in iou_types:
        # create a Coco-like object that we can use to evaluate detection!
        anns = []
        for image_id, gt in enumerate(groundtruths):
            labels = gt.get_field('labels').tolist()  # integer
            boxes = gt.bbox.tolist()  # xyxy
            for cls, box in zip(labels, boxes):
                anns.append({
                    'area': (box[3] - box[1] + 1) * (box[2] - box[0] + 1),
                    'bbox': [box[0], box[1], box[2] - box[0] + 1, box[3] - box[1] + 1],  # xywh
                    'category_id': cls,
                    'id': len(anns),
                    'image_id': image_id,
                    'iscrowd': 0,
                })
        fauxcoco = COCO()
        fauxcoco.dataset = {
            'info': {'description': 'use coco script for vg detection evaluation'},
            'images': [{'id': i} for i in range(len(groundtruths))],
            'categories': [
                {'supercategory': 'person', 'id': i, 'name': name}
                for i, name in enumerate(dataset.ind_to_classes) if name != '__background__'
            ],
            'annotations': anns,
        }
        fauxcoco.createIndex()

        # format predictions to coco-like
        cocolike_predictions = []
        for image_id, prediction in enumerate(predictions):
            box = prediction.convert('xywh').bbox.detach().cpu().numpy()  # xywh
            score = prediction.get_field('pred_scores').detach().cpu().numpy()  # (#objs,)
            label = prediction.get_field('pred_labels').detach().cpu().numpy()  # (#objs,)
            # for predcls, we set label and score to groundtruth
            if mode == 'predcls':
                label = prediction.get_field('labels').detach().cpu().numpy()
                score = np.ones(label.shape[0])
                assert len(label) == len(box)
            image_id = np.asarray([image_id] * len(box))
            cocolike_predictions.append(
                np.column_stack((image_id, box, score, label))
            )
            # logger.info(cocolike_predictions)
        cocolike_predictions = np.concatenate(cocolike_predictions, 0)

        # logger.info("Evaluating bbox proposals")
        # areas = {"all": "", "small": "s", "medium": "m", "large": "l"}
        # res = COCOResults("box_proposal")
        # for limit in [100, 1000]:
        #     for area, suffix in areas.items():
        #         stats = evaluate_box_proposals(
        #             predictions, dataset, area=area, limit=limit
        #         )
        #         key = "AR{}@{:d}".format(suffix, limit)
        #         res.results["box_proposal"][key] = stats["ar"].item()
        # logger.info(res)
        # if output_folder:
        #     torch.save(res, os.path.join(output_folder, "box_proposals.pth"))

        # evaluate via coco API
        res = fauxcoco.loadRes(cocolike_predictions)
        coco_eval = COCOeval(fauxcoco, res, 'bbox')
        coco_eval.params.imgIds = list(range(len(groundtruths)))
        coco_eval.evaluate()
        coco_eval.accumulate()
        coco_eval.summarize()
        coco_res = COCOResults('bbox')
        coco_res.update(coco_eval)
        mAp = coco_eval.stats[1]

        def get_coco_eval(coco_eval, iouThr, eval_type, maxDets=-1, areaRng="all"):
            p = coco_eval.params

            aind = [i for i, aRng in enumerate(p.areaRngLbl) if aRng == areaRng]
            if maxDets == -1:
                max_range_i = np.argmax(p.maxDets)
                mind = [max_range_i, ]
            else:
                mind = [i for i, mDet in enumerate(p.maxDets) if mDet == maxDets]

            if eval_type == 'precision':
                # dimension of precision: [TxRxKxAxM]
                s = coco_eval.eval['precision']
                # IoU
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, :, aind, mind]
            elif eval_type == 'recall':
                # dimension of recall: [TxKxAxM]
                s = coco_eval.eval['recall']
                if iouThr is not None:
                    t = np.where(iouThr == p.iouThrs)[0]
                    s = s[t]
                s = s[:, :, aind, mind]
            else:
                raise ValueError("Invalid eval metrics")
            if len(s[s > -1]) == 0:
                mean_s = -1
            else:
                mean_s = np.mean(s[s > -1])
            return p.maxDets[mind[-1]], mean_s

        coco_res_to_save = {}
        for key, value in coco_res.results.items():
            for evl_name, eval_val in value.items():
                coco_res_to_save[f"{key}/{evl_name}"] = eval_val
        result_dict_list_to_log.append(coco_res_to_save)

        result_str += 'Detection evaluation mAp=%.4f\n' % mAp
        result_str += "recall@%d IOU:0.5 %.4f\n" % get_coco_eval(coco_eval, 0.5, 'recall')
        result_str += '=' * 100 + '\n'
        avg_metrics = mAp
        logger.info(result_str)
        result_str = '\n'
        logger.info("box evaluation done!")

    if "relations" in iou_types:
        result_str = '\n'
        evaluator = {}
        rel_eval_result_dict = {}
        # tradictional Recall@K
        eval_recall = SGRecall(rel_eval_result_dict)
        eval_recall.register_container(mode)
        evaluator['eval_recall'] = eval_recall

        # no graphical constraint
        eval_nog_recall = SGNoGraphConstraintRecall(rel_eval_result_dict)
        eval_nog_recall.register_container(mode)
        evaluator['eval_nog_recall'] = eval_nog_recall

        # test on different distribution
        eval_zeroshot_recall = SGZeroShotRecall(rel_eval_result_dict)
        eval_zeroshot_recall.register_container(mode)
        evaluator['eval_zeroshot_recall'] = eval_zeroshot_recall

        # used by https://github.com/NVIDIA/ContrastiveLosses4VRD for sgcls and predcls
        eval_pair_accuracy = SGPairAccuracy(rel_eval_result_dict)
        eval_pair_accuracy.register_container(mode)
        evaluator['eval_pair_accuracy'] = eval_pair_accuracy

        # used for meanRecall@K
        eval_mean_recall = SGMeanRecall(rel_eval_result_dict, num_rel_category, dataset.ind_to_predicates,
                                        print_detail=True)
        eval_mean_recall.register_container(mode)
        evaluator['eval_mean_recall'] = eval_mean_recall

        # used for NG-meanRecall@K
        eval_ng_mean_recall = SGNGMeanRecall(result_dict, num_rel_category, dataset.ind_to_predicates,
                                             print_detail=True)
        eval_ng_mean_recall.register_container(mode)
        evaluator['eval_ng_mean_recall'] = eval_ng_mean_recall

        eval_stagewise_recall = SGStagewiseRecall(rel_eval_result_dict)
        eval_stagewise_recall.register_container(mode)
        evaluator['eval_stagewise_recall'] = eval_stagewise_recall

        # prepare all inputs
        global_container = {}
        global_container['zeroshot_triplet'] = zeroshot_triplet
        global_container['result_dict'] = rel_eval_result_dict
        global_container['mode'] = mode
        global_container['multiple_preds'] = multiple_preds
        global_container['num_rel_category'] = num_rel_category
        global_container['iou_thres'] = iou_thres
        global_container['attribute_on'] = attribute_on
        global_container['num_attributes'] = num_attributes

        logger.info("evaluating relationship predictions..")
        for groundtruth, prediction in tqdm(zip(groundtruths, predictions), total=len(predictions)):
            evaluate_relation_of_one_image(groundtruth, prediction, global_container, evaluator)

        # calculate mean recall
        eval_mean_recall.calculate_mean_recall(mode)
        eval_ng_mean_recall.calculate_mean_recall(mode)

        def generate_eval_res_dict(evaluator, mode):
            res_dict = {}
            for k, v in evaluator.result_dict[f'{mode}_{evaluator.type}'].items():
                res_dict[f'{mode}_{evaluator.type}/top{k}'] = np.mean(v)
            return res_dict

        def longtail_part_eval(evaluator, mode):
            longtail_part_dict = [None, 'b', 't', 't', 't', 'b', 'b', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 't', 'b',
                                't', 't', 'b', 'h', 'b', 'h', 'b', 't', 'b', 't', 't', 't', 'h', 'h', 'h', 't', 'b',
                                't', 'b', 't', 't', 'b', 't', 'b', 'b', 't', 'b', 't', 't', 'b', 'b', 'h', 'b', 'b']
            assert "mean_recall" in evaluator.type
            res_dict = {}
            res_str = "\nlongtail part recall:\n"
            for topk, cate_rec_list in evaluator.result_dict[f'{mode}_{evaluator.type}_list'].items():
                part_recall = {"h": [], "b": [], "t": [], }
                for idx, each_cat_recall in enumerate(cate_rec_list):
                    part_recall[longtail_part_dict[idx + 1]].append(each_cat_recall)
                res_dict[f"sgdet_longtail_part_recall/top{topk}/head"] = np.mean(part_recall['h'])
                res_dict[f"sgdet_longtail_part_recall/top{topk}/body"] = np.mean(part_recall['b'])
                res_dict[f"sgdet_longtail_part_recall/top{topk}/tail"] = np.mean(part_recall['t'])
                res_str += f"Top{topk:4}: head: {np.mean(part_recall['h']):.4f} " \
                           f"body: {np.mean(part_recall['b']):.4f} " \
                           f"tail: {np.mean(part_recall['t']):.4f}\n"

            return res_dict, res_str

        # show the distribution & recall_count
        pred_counter_dir = os.path.join(args.output_dir, "pred_counter.pkl")
        if os.path.exists(pred_counter_dir):
            with open(pred_counter_dir, 'rb') as f:
                pred_counter = pickle.load(f)

            def show_per_cls_performance_and_frequency(mean_recall_evaluator, per_cls_res_dict):
                cls_dict = mean_recall_evaluator.rel_name_list
                cate_recall = []
                cate_num = []
                cate_set = []
                counter_name = []
                for cate_set_idx, name_set in enumerate([HEAD, BODY, TAIL]):
                    for cate_id in name_set:
                        cate_set.append(cate_set_idx)
                        counter_name.append(cls_dict[cate_id - 1])  # list start from 0
                        cate_recall.append(per_cls_res_dict[cate_id - 1])  # list start from 0
                        cate_num.append(pred_counter[cate_id])  # dict start from 1

                def min_max_norm(data):
                    return (data - min(data)) / max(data)

                cate_num = min_max_norm(np.array(cate_num))
                cate_recall = np.array(cate_recall)
                # cate_recall = min_max_norm(np.array(cate_recall))

                fig, axs_c = plt.subplots(1, 1, figsize=(13, 5), tight_layout=True)
                pallte = ['r', 'g', 'b']
                color = [pallte[idx] for idx in cate_set]
                axs_c.bar(counter_name, cate_num, color=color, width=0.6, zorder=0)
                axs_c.scatter(counter_name, cate_recall, color='k', zorder=10)

                plt.xticks(rotation=-90, )
                axs_c.grid()
                fig.set_facecolor((1, 1, 1))

                global eval_times
                eval_times += 1
                save_file = os.path.join(args.output_dir,
                                         f"rel_freq_dist2recall-{mean_recall_evaluator.type}-{eval_times}.png")
                fig.savefig(save_file, dpi=300)

        per_cls_res_dict = eval_mean_recall.result_dict[f'{mode}_{eval_mean_recall.type}_list'][100]
        show_per_cls_performance_and_frequency(eval_mean_recall, per_cls_res_dict)

        per_cls_res_dict = eval_ng_mean_recall.result_dict[f'{mode}_{eval_ng_mean_recall.type}_list'][100]
        show_per_cls_performance_and_frequency(eval_ng_mean_recall, per_cls_res_dict)

        longtail_part_res_dict, longtail_part_res_str = longtail_part_eval(eval_mean_recall, mode)
        ng_longtail_part_res_dict, ng_longtail_part_res_str = longtail_part_eval(eval_ng_mean_recall, mode)
        
        # print result
        result_str += eval_recall.generate_print_string(mode)
        result_str += eval_nog_recall.generate_print_string(mode)
        result_str += eval_zeroshot_recall.generate_print_string(mode)
        result_str += eval_mean_recall.generate_print_string(mode)
        result_str += eval_ng_mean_recall.generate_print_string(mode)
        result_str += eval_stagewise_recall.generate_print_string(mode)
        result_str += longtail_part_res_str
        result_str += f"(Non-Graph-Constraint) {ng_longtail_part_res_str}"

        result_dict_list_to_log.extend([generate_eval_res_dict(eval_recall, mode),
                                        generate_eval_res_dict(eval_nog_recall, mode),
                                        generate_eval_res_dict(eval_zeroshot_recall, mode),
                                        generate_eval_res_dict(eval_mean_recall, mode),
                                        generate_eval_res_dict(eval_ng_mean_recall, mode),
                                        longtail_part_res_dict,ng_longtail_part_res_dict])

        # if cfg.MODEL.ROI_RELATION_HEAD.USE_GT_BOX:
        #     result_str += eval_pair_accuracy.generate_print_string(mode)

        result_str += '=' * 100 + '\n'
        # average the all recall and mean recall with the weight
        avg_metrics = np.mean(rel_eval_result_dict[mode + '_recall'][100]) * 0.5 \
                      + np.mean(rel_eval_result_dict[mode + '_mean_recall'][100]) * 0.5

        if output_folder:
            torch.save(rel_eval_result_dict, os.path.join(output_folder, 'result_dict.pytorch'))

    logger.info(result_str)
    if output_folder:
        with open(os.path.join(output_folder, "evaluation_res.txt"), 'w') as f:
            f.write(result_str)

    return float(avg_metrics), result_dict_list_to_log


def compute_on_dataset(model, data_loader, device, synchronize_gather=True, timer=None, logger=None):
    """

    :param model:
    :param data_loader:
    :param device:
    :param synchronize_gather:  gather the predictions during the training,
                                rather than gathering all predictions after the training
    :param timer:
    :return:
    """
    model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    for _, batch in enumerate(tqdm(data_loader)):
        with torch.no_grad():
            images, targets, image_ids = batch
            targets = [target.to(device) for target in targets]
            if timer:
                timer.tic()
            # if cfg.TEST.BBOX_AUG.ENABLED:
            #     output = im_detect_bbox_aug(model, images, device)
            # else:
                # relation detection needs the targets
            output = model(images.to(device), targets, logger=logger)
            if timer:
                if not device == 'cpu':
                    torch.cuda.synchronize()
                timer.toc()
            output = [o.to(cpu_device) for o in output]
        if synchronize_gather:
            synchronize()
            multi_gpu_predictions = all_gather({img_id: result for img_id, result in zip(image_ids, output)})
            if is_main_process():
                for p in multi_gpu_predictions:
                    results_dict.update(p)
        else:
            results_dict.update(
                {img_id: result for img_id, result in zip(image_ids, output)}
            )
    return results_dict

def inference(
        args,
        model,
        data_loader,
        dataset_name,
        iou_types=("bbox",),
        box_only=False,
        device="cuda",
        expected_results=(),
        expected_results_sigma_tol=4,
        output_folder=None,
        logger=None,
):
    load_prediction_from_cache = False and output_folder is not None and os.path.exists(
        os.path.join(output_folder, "eval_results.pytorch"))
    # convert to a torch.device for efficiency
    device = torch.device(device)
    num_devices = get_world_size()
    if logger is None:
        logger = logging.getLogger("pysgg.inference")
    dataset = data_loader.dataset
    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    if load_prediction_from_cache:
        logging.info("load_prediction_from_cache: " + os.path.join(output_folder, "eval_results.pytorch"))
        predictions = torch.load(os.path.join(output_folder, "eval_results.pytorch"),
                                 map_location=torch.device("cpu"))['predictions']
    else:
        predictions = compute_on_dataset(model, data_loader, device,
                                         synchronize_gather = True,
                                         timer=inference_timer, logger=logger)
    # wait for all processes to complete before measuring the time
    synchronize()
    total_time = total_timer.toc()
    total_time_str = get_time_str(total_time)
    logger.info(
        "Total run time: {} ({} s / img per device, on {} devices)".format(
            total_time_str, total_time * num_devices / len(dataset), num_devices
        )
    )
    total_infer_time = get_time_str(inference_timer.total_time)
    logger.info(
        "Model inference time: {} ({} s / img per device, on {} devices)".format(
            total_infer_time,
            inference_timer.total_time * num_devices / len(dataset),
            num_devices,
        )
    )

    if not load_prediction_from_cache:
        predictions = _accumulate_predictions_from_multiple_gpus(predictions,
                                                                 synchronize_gather=True)

    if not is_main_process():
        return -1.0

    # if output_folder is not None and not load_prediction_from_cache:
    #    torch.save(predictions, os.path.join(output_folder, "predictions.pth"))

    # extra_args = dict(
    #     box_only=box_only,
    #     iou_types=iou_types,
    #     expected_results=expected_results,
    #     expected_results_sigma_tol=expected_results_sigma_tol,
    # )
    return do_vg_evaluation(args=args,
                    dataset=dataset,
                    predictions=predictions,
                    output_folder=output_folder,
                    logger=logger,
                    iou_types=iou_types)


def run_val(args, model, val_data_loaders, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    # if cfg.MODEL.RELATION_ON:
    iou_types = iou_types + ("relations",)
    # if cfg.MODEL.ATTRIBUTE_ON:
    #     iou_types = iou_types + ("attributes",)

    dataset_names = [args.dataset_name_val]
    val_result = []
    for dataset_name, val_data_loader in zip(dataset_names, [val_data_loaders]):
        dataset_result = inference(
            args,
            model,
            val_data_loader,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False,
            device=args.device,
            expected_results=[],
            expected_results_sigma_tol=4,
            output_folder=None,
            logger=logger,
        )
        synchronize()
        val_result.append(dataset_result)

    val_values = []
    for each in val_result:
        if isinstance(each, tuple):
            val_values.append(each[0])
    # support for multi gpu distributed testing
    # send evaluation results to each process
    gathered_result = all_gather(torch.tensor(val_values).cpu())
    gathered_result = [t.view(-1) for t in gathered_result]
    gathered_result = torch.cat(gathered_result, dim=-1).view(-1)
    valid_result = gathered_result[gathered_result >= 0]
    val_result_val = float(valid_result.mean())

    del gathered_result, valid_result
    return val_result, val_result_val

def run_test(args, model, distributed, logger):
    if distributed:
        model = model.module
    iou_types = ("bbox",)
    # if cfg.MODEL.MASK_ON:
    #     iou_types = iou_types + ("segm",)
    # if cfg.MODEL.KEYPOINT_ON:
    #     iou_types = iou_types + ("keypoints",)
    # if cfg.MODEL.RELATION_ON:
    iou_types = iou_types + ("relations",)
    # if cfg.MODEL.ATTRIBUTE_ON:
    #   iou_types = iou_types + ("attributes",)
    output_folders = [None]
    dataset_names = [args.dataset_name_test]
    if args.output_dir:
        for idx, dataset_name in enumerate(dataset_names):
            output_folder = os.path.join(args.output_dir, "inference", dataset_name)
            mkdir(output_folder)
            output_folders[idx] = output_folder
    data_loaders_val = make_data_loader(args, mode="test", is_distributed=distributed)
    for output_folder, dataset_name, data_loader_val in zip(
        output_folders, dataset_names, [data_loaders_val]
    ):
        inference(
            args,
            model,
            data_loader_val,
            dataset_name=dataset_name,
            iou_types=iou_types,
            box_only=False,
            device=args.device,
            expected_results=[],
            expected_results_sigma_tol=4,
            output_folder=output_folder,
            logger=logger,
        )
        synchronize()
