# Originated from https://github.com/PAIR-code/saliency/blob/master/saliency/xrai.py

import numpy as np
import webp
from PIL import Image
from skimage import segmentation
from skimage.morphology import dilation, disk
from skimage.transform import resize
from webp import WebPConfig

_FELZENSZWALB_SCALE_VALUES = [50, 100, 150, 250, 500, 1200]
_FELZENSZWALB_SIGMA_VALUES = [0.8]
_FELZENSZWALB_IM_RESIZE = (224, 224)
_FELZENSZWALB_IM_VALUE_RANGE = [-1.0, 1.0]
_FELZENSZWALB_MIN_SEGMENT_SIZE = 150


def _normalize_image(im, value_range, resize_shape=None):
    """Normalize an image by resizing it and rescaling its values
    Args:
        im: Input image.
        value_range: [min_value, max_value]
        resize_shape: New image shape. Defaults to None.
    Returns:
        Resized and rescaled image.
    """
    im_max = np.max(im)
    im_min = np.min(im)
    im = (im - im_min) / (im_max - im_min)
    im = im * (value_range[1] - value_range[0]) + value_range[0]
    if resize_shape is not None:
        im = resize(im,
                    resize_shape,
                    order=3,
                    mode='constant',
                    preserve_range=True,
                    anti_aliasing=True)
    return im


def _get_segments_felzenszwalb(im,
                               resize_image=True,
                               scale_range=None,
                               dilation_rad=5):
    """Compute image segments based on Felzenszwalb's algorithm.
    Efficient graph-based image segmentation, Felzenszwalb, P.F.
    and Huttenlocher, D.P. International Journal of Computer Vision, 2004
    Args:
      im: Input image.
      resize_image: If True, the image is resized to 224,224 for the segmentation
                    purposes. The resulting segments are rescaled back to match
                    the original image size. It is done for consistency w.r.t.
                    segmentation parameter range. Defaults to True.
      scale_range:  Range of image values to use for segmentation algorithm.
                    Segmentation algorithm is sensitive to the input image
                    values, therefore we need to be consistent with the range
                    for all images. If None is passed, the range is scaled to
                    [-1.0, 1.0]. Defaults to None.
      dilation_rad: Sets how much each segment is dilated to include edges,
                    larger values cause more blobby segments, smaller values
                    get sharper areas. Defaults to 5.
    Returns:
        masks: A list of boolean masks as np.ndarrays if size HxW for im size of
               HxWxC.
    """

    # TODO (tolgab) Set this to default float range of 0.0 - 1.0 and tune
    # parameters for that
    if scale_range is None:
        scale_range = _FELZENSZWALB_IM_VALUE_RANGE
    # Normalize image value range and size
    original_shape = im.shape[:2]
    # TODO (tolgab) This resize is unnecessary with more intelligent param range
    # selection
    if resize_image:
        im = _normalize_image(im, scale_range, _FELZENSZWALB_IM_RESIZE)
    else:
        im = _normalize_image(im, scale_range)
    segs = []
    for scale in _FELZENSZWALB_SCALE_VALUES:
        for sigma in _FELZENSZWALB_SIGMA_VALUES:
            seg = segmentation.felzenszwalb(im,
                                            scale=scale,
                                            sigma=sigma,
                                            min_size=_FELZENSZWALB_MIN_SEGMENT_SIZE)
            if resize_image:
                seg = resize(seg,
                             original_shape,
                             order=0,
                             preserve_range=True,
                             mode='constant',
                             anti_aliasing=False).astype(np.int)
            segs.append(seg)
    masks = _unpack_segs_to_masks(segs)
    if dilation_rad:
        selem = disk(dilation_rad)
        masks = [dilation(mask, selem=selem) for mask in masks]
    return masks


def _attr_aggregation_max(attr, axis=-1):
    return attr.max(axis=axis)


def _gain_density(mask1, attr, mask2=None):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _get_diff_mask(add_mask, base_mask):
    return np.logical_and(add_mask, np.logical_not(base_mask))


def _get_diff_cnt(add_mask, base_mask):
    return np.sum(_get_diff_mask(add_mask, base_mask))


def _unpack_segs_to_masks(segs):
    masks = []
    for seg in segs:
        for l in range(seg.min(), seg.max() + 1):
            masks.append(seg == l)
    return masks


def _gain_density(mask1, attr, mask2=None):
    # Compute the attr density over mask1. If mask2 is specified, compute density
    # for mask1 \ mask2
    if mask2 is None:
        added_mask = mask1
    else:
        added_mask = _get_diff_mask(mask1, mask2)
    if not np.any(added_mask):
        return -np.inf
    else:
        return attr[added_mask].mean()


def _xrai(attr,
          segs,
          gain_fun=_gain_density,
          area_perc_th=1.0,
          min_pixel_diff=50,
          integer_segments=True):
    """Run XRAI saliency given attributions and segments.
    Args:
        attr: Source attributions for XRAI. XRAI attributions will be same size
                as the input attr.
        segs: Input segments as a list of boolean masks. XRAI uses these to
                compute attribution sums.
        gain_fun: The function that computes XRAI area attribution from source
                    attributions. Defaults to _gain_density, which calculates the
                    density of attributions in a mask.
        area_perc_th: The saliency map is computed to cover area_perc_th of
                        the image. Lower values will run faster, but produce
                        uncomputed areas in the image that will be filled to
                        satisfy completeness. Defaults to 1.0.
        min_pixel_diff: Do not consider masks that have difference less than
                        this number compared to the current mask. Set it to 1
                        to remove masks that completely overlap with the
                        current mask.
        integer_segments: See XRAIParameters. Defaults to True.
    Returns:
        tuple: saliency heatmap and list of masks or an integer image with
                area ranks depending on the parameter integer_segments.
    """
    output_attr = -np.inf * np.ones(shape=attr.shape, dtype=np.float)

    n_masks = len(segs)
    current_area_perc = 0.0
    current_mask = np.zeros(attr.shape, dtype=bool)

    masks_trace = []
    remaining_masks = {ind: mask for ind, mask in enumerate(segs)}

    added_masks_cnt = 1
    # While the mask area is less than area_th and remaining_masks is not empty
    while current_area_perc <= area_perc_th:
        best_gain = -np.inf
        best_key = None
        remove_key_queue = []
        for mask_key in remaining_masks:
            mask = remaining_masks[mask_key]
            # If mask does not add more than min_pixel_diff to current mask, remove
            mask_pixel_diff = _get_diff_cnt(mask, current_mask)
            if mask_pixel_diff < min_pixel_diff:
                remove_key_queue.append(mask_key)
                continue
            gain = gain_fun(mask, attr, mask2=current_mask)
            if gain > best_gain:
                best_gain = gain
                best_key = mask_key
        for key in remove_key_queue:
            del remaining_masks[key]
        if len(remaining_masks) == 0:
            break
        added_mask = remaining_masks[best_key]
        mask_diff = _get_diff_mask(added_mask, current_mask)
        masks_trace.append((mask_diff, best_gain))

        current_mask = np.logical_or(current_mask, added_mask)
        current_area_perc = np.mean(current_mask)
        output_attr[mask_diff] = best_gain
        del remaining_masks[best_key]  # delete used key
        added_masks_cnt += 1

    uncomputed_mask = output_attr == -np.inf
    # Assign the uncomputed areas a value such that sum is same as ig
    output_attr[uncomputed_mask] = gain_fun(uncomputed_mask, attr)
    masks_trace = [v[0] for v in sorted(masks_trace, key=lambda x: -x[1])]
    if np.any(uncomputed_mask):
        masks_trace.append(uncomputed_mask)
    if integer_segments:
        attr_ranks = np.zeros(shape=attr.shape, dtype=np.int)
        for i, mask in enumerate(masks_trace):
            attr_ranks[mask] = i + 1
        return output_attr, attr_ranks
    else:
        return output_attr, masks_trace


def get_webp_length(np_image: np.ndarray, debug: int = 0) -> int:
    PIL_image = Image.fromarray(np_image)
    pic = webp.WebPPicture.from_pil(PIL_image)
    config = WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=70)
    buf = pic.encode(config).buffer()
    length = len(buf)

    if debug > 0:
        pic.save(f'test_{debug}.webp')

    return length


if __name__ == "__main__":
    from vision_xai_tools.datasets import ImageNetDataset

    dataset = ImageNetDataset('valid', normalize=None)
    input, target, fname = dataset[0]

    from PIL import Image

    im = input.permute(1, 2, 0).numpy()

    import webp
    from webp import WebPConfig

    segs = _get_segments_felzenszwalb(im)

    attr = np.random.normal(loc=0, scale=1, size=[224, 224])

    output_attr, masks_trace = _xrai(attr, segs, integer_segments=False)

    PIL_image = Image.fromarray(np.uint8(im * 255.))
    pic = webp.WebPPicture.from_pil(PIL_image)
    config = WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=70)
    buf = pic.encode(config).buffer()

    full_length = len(buf)
    print(full_length)

    new_im = np.uint8(im * 255.)
    mean = new_im.mean(0).mean(0)
    new_im = mean.astype(np.uint8).reshape(
        1, 1, 3).repeat(224, 0).repeat(224, 1)

    idx = 5

    for mask in masks_trace:
        new_im[mask] = np.uint8(im * 255.)[mask]

        PIL_image = Image.fromarray(new_im)
        pic = webp.WebPPicture.from_pil(PIL_image)
        config = WebPConfig.new(preset=webp.WebPPreset.PHOTO, quality=70)
        buf = pic.encode(config).buffer()
        length = len(buf)

        percentage = length / full_length

        if int(percentage * 100) >= idx:
            print(percentage)
            idx += 5

            if percentage >= 1.:
                break

    PIL_image.save('test.webp')

    pass
