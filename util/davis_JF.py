# ----------------------------------------------------------------------------
# The 2017 DAVIS Challenge on Video Object Segmentation
#-----------------------------------------------------------------------------
# Copyright (c) 2017 Federico Perazzi
# Licensed under the BSD License [see LICENSE for details]
# Written by Federico Perazzi (federico@disneyresearch.com)
# Adapted from DAVIS 2016 (Federico Perazzi)
# ----------------------------------------------------------------------------

import numpy as np
import math
import cv2
""" Utilities for computing, reading and saving benchmark evaluation."""

def db_eval_boundary(foreground_mask,gt_mask,bound_th=0.010):
	"""
	Compute mean,recall and decay from per-frame evaluation.
	Calculates precision/recall for boundaries between foreground_mask and
	gt_mask using morphological operators to speed it up.
	Arguments:
		foreground_mask (ndarray): binary segmentation image.
		gt_mask         (ndarray): binary annotated image.
	Returns:
		F (float): boundaries F-measure
		P (float): boundaries precision
		R (float): boundaries recall
	"""
	assert np.atleast_3d(foreground_mask).shape[2] == 1

	bound_pix = bound_th if bound_th >= 1 else \
			np.ceil(bound_th*np.linalg.norm(foreground_mask.shape))

	# Get the pixel boundaries of both masks
	fg_boundary = seg2bmap(foreground_mask)
	gt_boundary = seg2bmap(gt_mask)

	from skimage.morphology import binary_dilation,disk

	# fg_dil = binary_dilation(fg_boundary,disk(bound_pix))
	# gt_dil = binary_dilation(gt_boundary,disk(bound_pix))

	fg_dil = cv2.dilate(fg_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
	gt_dil = cv2.dilate(gt_boundary.astype(np.uint8), disk(bound_pix).astype(np.uint8))
	# Get the intersection
	gt_match = gt_boundary * fg_dil
	fg_match = fg_boundary * gt_dil

	# Area of the intersection
	n_fg     = np.sum(fg_boundary)
	n_gt     = np.sum(gt_boundary)

	#% Compute precision and recall
	if n_fg == 0 and  n_gt > 0:
		precision = 1
		recall = 0
	elif n_fg > 0 and n_gt == 0:
		precision = 0
		recall = 1
	elif n_fg == 0  and n_gt == 0:
		precision = 1
		recall = 1
	else:
		precision = np.sum(fg_match)/float(n_fg)
		recall    = np.sum(gt_match)/float(n_gt)

	# Compute F measure
	if precision + recall == 0:
		F = 0
	else:
		F = 2*precision*recall/(precision+recall)

	return F

def seg2bmap(seg,width=None,height=None):
	"""
	From a segmentation, compute a binary boundary map with 1 pixel wide
	boundaries.  The boundary pixels are offset by 1/2 pixel towards the
	origin from the actual segment boundary.
	Arguments:
		seg     : Segments labeled from 1..k.
		width	  :	Width of desired bmap  <= seg.shape[1]
		height  :	Height of desired bmap <= seg.shape[0]
	Returns:
		bmap (ndarray):	Binary boundary map.
	 David Martin <dmartin@eecs.berkeley.edu>
	 January 2003
 """

	seg = seg.astype(np.bool)
	seg[seg>0] = 1

	assert np.atleast_3d(seg).shape[2] == 1

	width  = seg.shape[1] if width  is None else width
	height = seg.shape[0] if height is None else height

	h,w = seg.shape[:2]

	ar1 = float(width) / float(height)
	ar2 = float(w) / float(h)

	assert not (width>w | height>h | abs(ar1-ar2)>0.01),\
			'Can''t convert %dx%d seg to %dx%d bmap.'%(w,h,width,height)

	e  = np.zeros_like(seg)
	s  = np.zeros_like(seg)
	se = np.zeros_like(seg)

	e[:,:-1]    = seg[:,1:]
	s[:-1,:]    = seg[1:,:]
	se[:-1,:-1] = seg[1:,1:]

	b        = seg^e | seg^s | seg^se
	b[-1,:]  = seg[-1,:]^e[-1,:]
	b[:,-1]  = seg[:,-1]^s[:,-1]
	b[-1,-1] = 0

	if w == width and h == height:
		bmap = b
	else:
		bmap = np.zeros((height,width))
		for x in range(w):
			for y in range(h):
				if b[y,x]:
					j = 1+math.floor((y-1)+height / h)
					i = 1+math.floor((x-1)+width  / h)
					bmap[j,i] = 1

	return bmap

def db_eval_iou(annotation,segmentation,void_pixels=None):

	""" Compute region similarity as the Jaccard Index.
	Arguments:
		annotation   (ndarray): binary annotation   map.
		segmentation (ndarray): binary segmentation map.
	Return:
		jaccard (float): region similarity
 """

	annotation   = annotation.astype(np.bool)
	segmentation = segmentation.astype(np.bool)

	if np.isclose(np.sum(annotation),0) and np.isclose(np.sum(segmentation),0):
		return 1
	else:
		return np.sum((annotation & segmentation)) / \
				np.sum((annotation | segmentation),dtype=np.float32)



	# assert annotation.shape == segmentation.shape, \
	#     f'Annotation({annotation.shape}) and segmentation:{segmentation.shape} dimensions do not match.'
	# annotation = annotation.astype(np.bool)
	# segmentation = segmentation.astype(np.bool)
	#
	# if void_pixels is not None:
	#     assert annotation.shape == void_pixels.shape, \
	#         f'Annotation({annotation.shape}) and void pixels:{void_pixels.shape} dimensions do not match.'
	#     void_pixels = void_pixels.astype(np.bool)
	# else:
	#     void_pixels = np.zeros_like(segmentation)
	#
	# # Intersection between all sets
	# inters = np.sum((segmentation & annotation) & np.logical_not(void_pixels), axis=(-2, -1))
	# union = np.sum((segmentation | annotation) & np.logical_not(void_pixels), axis=(-2, -1))
	#
	# j = inters / union
	# if j.ndim == 0:
	#     j = 1 if np.isclose(union, 0) else j
	# else:
	#     j[np.isclose(union, 0)] = 1
	# return j