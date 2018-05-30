from deepnets import convprops as CP

vgg_keys = ', '.join([key for key in CP.VGG_MODELS])
CONV_MODEL_KEY_ERROR = {
    'vgg': f"ERROR: Key error. Possible keys are '{vgg_keys}'"
}
