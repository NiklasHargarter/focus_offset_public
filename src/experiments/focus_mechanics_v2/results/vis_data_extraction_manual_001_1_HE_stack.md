# Visual Study: Multi-Scale Context and Legacy Resolution Gap

## 1. Context Boundaries (Relative to Slide)
This image shows how much physical area on the slide is contained within the various 'Scales' while maintaining the center coordinate.

![Context Boxes](vis_context_boxes_001_1_HE_stack.png)

## 2. Input Scaling (Fixed Model View)
Compare the visual information provided to the model at each scale. Higher downscales provide more structural context at the cost of high-frequency detail.

![Model Inputs](vis_context_inputs_001_1_HE_stack.png)

## 3. The Legacy Resolution Gap
The legacy approach prioritized speed by reading low-resolution data for Brenner focus scores. This visual demonstrates the literal information density difference between the 1x Training patch and the 8x Legacy Brenner patch used for ground-truth labeling:

![Legacy Blur](vis_legacy_blur_001_1_HE_stack.png)

**Conclusion**: The 28x28 pixel patch (right) lacks the structural detail to precisely identify the focus peak of a high-resolution 60x sample. This explains the ~0.8 slice jitter found in our MAE benchmarks.