"""Instance-only fix for the verified Transformers 5.13.0 OWLv2 resize branch."""
import inspect
import textwrap
from types import MethodType


def repair_no_resize(image_processor):
    if getattr(image_processor, "_portable_no_resize_repaired", False):
        return
    import transformers

    if transformers.__version__ == "4.50.0":
        original = type(image_processor).preprocess
        source = textwrap.dedent(inspect.getsource(original))
        if (original.__module__ != "transformers.models.owlv2.image_processing_owlv2" or
                "if do_resize:\n        images = [" not in source):
            raise RuntimeError("Unexpected OWLv2 4.50.0 preprocessing source")
        print("OWLv2 4.50.0 already preserves images when do_resize=False", flush=True)
        return

    original = type(image_processor)._preprocess
    if (transformers.__version__ != "5.13.0" or original.__module__ !=
            "transformers.models.owlv2.image_processing_owlv2"):
        raise RuntimeError("OWLv2 no-resize repair requires the inspected Transformers 5.13.0 processor")
    source = textwrap.dedent(inspect.getsource(original))
    old = '''        if do_resize:
            resized_stack = self.resize(image=stacked_images, size=size, resample=resample)
            resized_images_grouped[shape] = resized_stack'''
    new = '''        if do_resize:
            resized_stack = self.resize(image=stacked_images, size=size, resample=resample)
        else:
            resized_stack = stacked_images
        resized_images_grouped[shape] = resized_stack'''
    if source.count(old) != 1:
        raise RuntimeError("Unexpected OWLv2 preprocessing source; refusing to patch")
    namespace = dict(original.__globals__)
    exec(compile(source.replace(old, new), "<portable-owlv2-no-resize>", "exec"), namespace)
    image_processor._preprocess = MethodType(namespace[original.__name__], image_processor)
    image_processor._portable_no_resize_repaired = True
    print("OWLv2 compatibility: instance-only do_resize=False passthrough applied", flush=True)
