#include <torch/extension.h>
#include "nms.h"
#include "ROIAlign.h"
#include "ROIPool.h"
#include "SigmoidFocalLoss.h"
#include "deform_conv.h"
#include "deform_pool.h"

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
      m.def("nms", &nms, "Non-maximum suppression (NMS)",
            py::arg("dets"), py::arg("scores"), py::arg("threshold"));

      m.def("roi_align_forward", &ROIAlign_forward, "Forward pass of ROI Align",
            py::arg("input"), py::arg("rois"), py::arg("spatial_scale"),
            py::arg("pooled_height"), py::arg("pooled_width"),
            py::arg("sampling_ratio"));

      m.def("roi_align_backward", &ROIAlign_backward, "Backward pass of ROI Align",
            py::arg("grad"), py::arg("rois"), py::arg("spatial_scale"),
            py::arg("pooled_height"), py::arg("pooled_width"),
            py::arg("batch_size"), py::arg("channels"), py::arg("height"), py::arg("width"),
            py::arg("sampling_ratio"));

      m.def("roi_pool_forward", &ROIPool_forward, "Forward pass of ROI Pooling",
            py::arg("input"), py::arg("rois"), py::arg("spatial_scale"),
            py::arg("pooled_height"), py::arg("pooled_width"));

      m.def("roi_pool_backward", &ROIPool_backward, "Backward pass of ROI Pooling",
            py::arg("grad"), py::arg("input"), py::arg("rois"), py::arg("argmax"),
            py::arg("spatial_scale"), py::arg("pooled_height"), py::arg("pooled_width"),
            py::arg("batch_size"), py::arg("channels"), py::arg("height"), py::arg("width"));

      m.def("sigmoid_focalloss_forward", &SigmoidFocalLoss_forward, "Forward pass of Sigmoid Focal Loss",
            py::arg("logits"), py::arg("targets"),py::arg("num_classes"),
            py::arg("gamma"), py::arg("alpha"));

      m.def("sigmoid_focalloss_backward", &SigmoidFocalLoss_backward, "Backward pass of Sigmoid Focal Loss",
            py::arg("logits"), py::arg("targets"),py::arg("d_losses"),py::arg("num_classes"),
            py::arg("gamma"), py::arg("alpha"));

      // Deformable Convolution (DCNv2)
      m.def("deform_conv_forward", &deform_conv_forward, "Forward pass of Deformable Convolution",
            py::arg("input"), py::arg("weight"), py::arg("offset"), py::arg("output"),
            py::arg("columns"), py::arg("ones"), py::arg("kW"),py::arg("kH"),
            py::arg("dW"),py::arg("dH"),py::arg("padW"),py::arg("padH"),
            py::arg("dilationW"),py::arg("dilationH"),  
            py::arg("group"), py::arg("deformable_group"),
            py::arg("im2col_step"));

      m.def("deform_conv_backward_input", &deform_conv_backward_input, "Backward pass for input of Deformable Convolution",
            py::arg("input"), py::arg("offset"), py::arg("gradOutput"),
            py::arg("gradIutput"),py::arg("gradOffset"),py::arg("weight"),
            py::arg("columns"), py::arg("kW"),py::arg("kH"),
            py::arg("dW"),py::arg("dH"),py::arg("padW"),py::arg("padH"),
            py::arg("dilationW"),py::arg("dilationH"),  
            py::arg("group"), py::arg("deformable_group"),
            py::arg("im2col_step"));

      m.def("deform_conv_backward_parameters", &deform_conv_backward_parameters, "Backward pass for parameters of Deformable Convolution",
            py::arg("input"), py::arg("offset"),py::arg("gradOutput"),py::arg("gradWeight"),
            py::arg("columns"),py::arg("ones"), py::arg("kW"),py::arg("kH"),
            py::arg("dW"),py::arg("dH"),py::arg("padW"),py::arg("padH"),
            py::arg("dilationW"),py::arg("dilationH"),  
            py::arg("group"), py::arg("deformable_group"),py::arg("scale"),
            py::arg("im2col_step"));

      m.def("modulated_deform_conv_forward", &modulated_deform_conv_forward, "Forward pass of Modulated Deformable Convolution",
            py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("ones"), py::arg("offset"),
            py::arg("mask"), py::arg("output"), py::arg("columns"),
            py::arg("kernel_h"),py::arg("kernel_w"), 
            py::arg("stride_h"), py::arg("stride_w"), 
            py::arg("pad_h"), py::arg("pad_w"), 
            py::arg("dilation_h"), py::arg("dilation_w"),   
            py::arg("group"), py::arg("deformable_group"),
            py::arg("with_bias"));

      m.def("modulated_deform_conv_backward", &modulated_deform_conv_backward, "Backward pass of Modulated Deformable Convolution",
            py::arg("input"), py::arg("weight"), py::arg("bias"), py::arg("ones"), py::arg("offset"),
            py::arg("mask"), py::arg("columns"), py::arg("grad_input"), py::arg("grad_weight"), 
            py::arg("grad_bias"), py::arg("grad_offset"), py::arg("grad_mask"), py::arg("grad_output"), 
            py::arg("kernel_h"),py::arg("kernel_w"), 
            py::arg("stride_h"), py::arg("stride_w"), 
            py::arg("pad_h"), py::arg("pad_w"), 
            py::arg("dilation_h"), py::arg("dilation_w"),   
            py::arg("group"), py::arg("deformable_group"),
            py::arg("with_bias"));

      m.def("deform_psroi_pooling_forward", &deform_psroi_pooling_forward, "Forward pass of Deformable PSROI Pooling",
            py::arg("input"), py::arg("bbox"), py::arg("trans"), py::arg("out"),
            py::arg("top_count"), py::arg("no_trans"),
            py::arg("spatial_scale"),py::arg("output_dim"), py::arg("group_size"),
            py::arg("pooled_size"), py::arg("part_size"), py::arg("sample_per_part"), 
            py::arg("trans_std"));

      m.def("deform_psroi_pooling_backward", &deform_psroi_pooling_backward, "Backward pass of Deformable PSROI Pooling",
            py::arg("out_grad"), py::arg("input"), py::arg("bbox"), py::arg("trans"), 
            py::arg("top_count"),py::arg("input_grad"), py::arg("trans_grad"),  
            py::arg("no_trans"),py::arg("spatial_scale"),
            py::arg("output_dim"), py::arg("group_size"),
            py::arg("pooled_size"), py::arg("part_size"), 
            py::arg("sample_per_part"), py::arg("trans_std"));
}
