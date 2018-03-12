"""Image Creator in python.

Note: This is not a wrapper of OpenMS's ImageCreator tool."""

from pyopenms import MzMLFile, FeatureXMLFile, FeatureMap, MSExperiment
from pyopenms import BilinearInterpolation
import numpy as np
import math
from PIL import Image
import h5py

def _create_spec_dict(exp):
    spectrum_dict = {}

    # iterating over all spectra
    for i, spec in enumerate(exp.getSpectra()):
        rt = spec.getRT()
        temp_int = []
        temp_mz = []

        # iterating over all peaks
        for peak in spec:
            temp_int.append(peak.getIntensity())
            temp_mz.append(peak.getMZ())
        dict_i = {'mz': temp_mz,
                  'intensity': temp_int}
        spectrum_dict[rt] = dict_i

    return spectrum_dict


def _create_feat_dict(feat_map):
    feature_dict = {}
    for i, feature in enumerate(feat_map):
        mz = feature.getMZ()
        rt = feature.getRT()
        intensity = feature.getIntensity()
        charge = feature.getCharge()
        quality = feature.getOverallQuality()
        convex_hulls = feature.getConvexHulls()
        min_y, min_x = convex_hulls[0].getBoundingBox().minPosition()
        max_y, max_x = convex_hulls[0].getBoundingBox().maxPosition()
        for cv in convex_hulls:
            min_vals = cv.getBoundingBox().minPosition()
            max_vals = cv.getBoundingBox().maxPosition()
            if min_y > min_vals[0]:
                min_y = min_vals[0]
            if min_x > min_vals[1]:
                min_x = min_vals[1]
            if max_y < max_vals[0]:
                max_y = max_vals[0]
            if max_x < max_vals[1]:
                max_x = max_vals[1]
        dict_i = {'mz': mz,
                  'intensity': intensity,
                  'charge': charge,
                  'quality': quality,
                  'min_x': min_x,
                  'min_y': min_y,
                  'max_x': max_x,
                  'max_y': max_y}
        feature_dict[rt] = dict_i
    return feature_dict


def _interpolation(exp):
    bilip = BilinearInterpolation()

    rows = exp.size()
    cols = int(math.ceil(exp.getMaxMZ() - exp.getMinMZ()))

    tmp = bilip.getData()
    tmp.resize(rows, cols, float())
    bilip.setData(tmp)

    # scans run bottom-up:
    bilip.setMapping_0(0.0,
                       exp.getMaxRT(),
                       float(rows - 1),
                       exp.getMinRT())
    # peaks run left-right:
    bilip.setMapping_1(0.0,
                       exp.getMinMZ(),
                       float(cols - 1),
                       exp.getMaxMZ())

    for spec in exp.getSpectra():
        mz_arr, intensity_arr = spec.get_peaks()
        for i in range(len(mz_arr)):
            bilip.addValue(spec.getRT(), mz_arr[i], intensity_arr[i])

    return bilip


def _create_data_array(rt_min, rt_max, mz_min, mz_max,
                       height, width):
    rt_values = np.linspace(rt_min, rt_max, num=height, endpoint=True,
                            dtype=np.float32)
    mz_values = np.linspace(mz_min, mz_max, num=width, endpoint=True,
                            dtype=np.float32)

    rows = len(rt_values) + 1    # rt values
    cols = len(mz_values) + 1    # mz values

    data_arr = np.zeros((rows, cols), dtype=np.float32)

    # first row stores all discrete MZ values
    data_arr[0, 1:] = mz_values
    # first column stores all discrete RT values
    data_arr[1:, 0] = rt_values

    return data_arr


def _fill_spec_array(spec_arr, bilip, spectrum_dict):
    for rt in spectrum_dict.keys():
        mz_list = spectrum_dict[rt]['mz']
        intensity_list = spectrum_dict[rt]['intensity']

        row_i = find_nearest(spec_arr[0:, 0], rt)
        for i, mz in enumerate(mz_list):
            col_i = find_nearest(spec_arr[0, 0:], mz)

            # spec_arr[row_i, col_i] = intensity_list[i]
            spec_arr[row_i, col_i] = bilip.value(rt, mz)
    """
    # Excessing Interpolation
    for i, rt in enumerate(spec_arr[1:, 0]):
        for j, mz in enumerate(spec_arr[0, 1:]):
	    spec_arr[i+1, j+1] = bilip.value(rt, mz)
    return spec_arr
    """


def _save_spectrum_image(spec_arr, name):
    rows, cols = spec_arr[1:, 1:].shape
    spec_img = Image.new(mode='L',
                         size=(cols, rows),
                         color="white")
    pixels = spec_img.load()

    # log-intensity
    #    for i in range(spec_arr.shape[0]):
    #        for j in range(spec_arr.shape[1]):
    #            if spec_arr[i+1, j+1] > 0:
    #                spec_arr[i+1, j+1] = math.log(spec_arr[i+1, j+1])
    #    spec_arr[1:, 1:] *= 255.0 / spec_arr[1:, 1:].max()  # normalize
    factor = spec_arr[1:, 1:].max()
    factor = math.log(factor)
    factor /= 100.0

    for i in range(cols):
        for j in range(rows):
            pixels[i, j] = int(255.0 - (spec_arr[j+1, i+1] / factor))
    spec_img.save(name)
    return spec_arr


def find_nearest(array, value):
    '''Returns indices of elements smaller and larger than the value.

    Since value is generated using np.arange(), the min and max value will not
    be checked.'''
    idx = np.searchsorted(array, value, side="right")
    if idx >= len(array):
        return len(array) - 1
    return idx - 1


def _insert_dot_values(feature_dict, bbox_arr):
    for rt in feature_dict.keys():
        mz = feature_dict[rt]['mz']

        row_i = find_nearest(bbox_arr[0:, 0], rt)
        col_i = find_nearest(bbox_arr[0, 0:], mz)

        # actual dot position
        bbox_arr[row_i, col_i] = 255

    return bbox_arr


def _insert_rect_values(feature_dict, bbox_arr):
    for rt in feature_dict.keys():
        # draw one pixel radius around label dot
        mz = feature_dict[rt]['mz']
        row_i = find_nearest(bbox_arr[0:, 0], rt)
        col_i = find_nearest(bbox_arr[0, 0:], mz)

        if row_i - 1 > 0:
            bbox_arr[row_i - 1, col_i] = 255
        if row_i + 1 < bbox_arr.shape[0]:
            bbox_arr[row_i + 1, col_i] = 255
        if col_i - 1 > 0:
            bbox_arr[row_i, col_i - 1] = 255
        if col_i + 1 < bbox_arr.shape[1]:
            bbox_arr[row_i, col_i + 1] = 255

        if row_i - 1 > 0 and col_i - 1 > 0:
            bbox_arr[row_i - 1, col_i - 1] = 128
        if row_i + 1 < bbox_arr.shape[0] and col_i - 1 > 0:
            bbox_arr[row_i + 1, col_i - 1] = 128
        if row_i - 1 > 0 and col_i + 1 < bbox_arr.shape[1]:
            bbox_arr[row_i - 1, col_i + 1] = 128
        if row_i + 1 < bbox_arr.shape[0] and col_i + 1 < bbox_arr.shape[1]:
            bbox_arr[row_i + 1, col_i + 1] = 128

        # bounding box
        draw_lower = True
        draw_upper = True
        draw_left = True
        draw_right = True

        # x's are mz-values, y's are rt-values
        x1 = feature_dict[rt]['min_x']
        x2 = feature_dict[rt]['max_x']
        y1 = feature_dict[rt]['min_y']
        y2 = feature_dict[rt]['max_y']

        # update rectangle coordinates according to image boundaries
        # if a coordinate is updated, don't draw the corresponding boundary
        if x1 < bbox_arr[0, 1]:  # min mz
            x1 = bbox_arr[0, 1]
            draw_left = False
        if x2 > bbox_arr[0, -1]:  # max mz
            x2 = bbox_arr[0, -1]
            draw_right = False
        if y1 < bbox_arr[1, 0]:  # min rt
            y1 = bbox_arr[1, 0]
            draw_lower = False
        if y2 > bbox_arr[-1, 0]:  # max rt
            y2 = bbox_arr[-1, 0]
            draw_upper = False

        col_x1 = find_nearest(bbox_arr[0, 0:], x1)
        col_x2 = find_nearest(bbox_arr[0, 0:], x2)
        row_y1 = find_nearest(bbox_arr[0:, 0], y1)
        row_y2 = find_nearest(bbox_arr[0:, 0], y2)

        # first vertical line
        if draw_left:
            for i in range(row_y1, row_y2 + 1):
                bbox_arr[i, col_x1] = 255

        # second vertical line
        if draw_right:
            for i in range(row_y1, row_y2 + 1):
                bbox_arr[i, col_x2] = 255

        # first horizontal line
        if draw_lower:
            for i in range(col_x1, col_x2 + 1):
                bbox_arr[row_y1, i] = 255

        # second horizontal line
        if draw_upper:
            for i in range(col_x1, col_x2 + 1):
                bbox_arr[row_y2, i] = 255

    return bbox_arr


def _save_labels_image(dot_arr, name):
    rows, cols = dot_arr[1:, 1:].shape
    labels_img = Image.new(mode='L',
                           size=(cols, rows),
                           color='white')
    pixels = labels_img.load()
    for i in range(cols):
        for j in range(rows):
            pixels[i, j] = int(255.0 - dot_arr[j+1, i+1])
    labels_img.save(name)
    return dot_arr


def _save_features_image(bbox_arr, name):
    rows, cols = bbox_arr[1:, 1:].shape
    feat_img = Image.new(mode='L',
                         size=(cols, rows),
                         color='white')
    pixels = feat_img.load()
    for i in range(cols):
        for j in range(rows):
            pixels[i, j] = int(255.0 - bbox_arr[j+1, i+1])
    feat_img.save(name)
    return bbox_arr


def _create_bbox_vectors(feature_dict, dot_arr, mz_step_size, rt_step_size):
    # create array of bbox pixel (x, y, left, right, up, down) values
    n_feat = len(feature_dict.keys())
    arr_shape = dot_arr.shape
    bbox_val = np.zeros((4, arr_shape[0]-1, arr_shape[1]-1), dtype=np.int32)
    
    for i, rt in enumerate(feature_dict.keys()):
        mz = feature_dict[rt]['mz']

        # dot_arr is already flipped vertically
        row_i = find_nearest(dot_arr[0:, 0], rt)
        col_i = find_nearest(dot_arr[0, 0:], mz)

        row_i = max(0, row_i - 1)
        col_i = max(0, col_i - 1)

        # x's are mz-values, y's are rt-values
        x1 = feature_dict[rt]['min_x']
        x2 = feature_dict[rt]['max_x']
        y1 = feature_dict[rt]['min_y']
        y2 = feature_dict[rt]['max_y']

        bbox_val[0, row_i, col_i] = np.rint((y2 - rt)/rt_step_size)  # pixels up
        bbox_val[1, row_i, col_i] = np.rint((x2 - mz)/mz_step_size)  # pixels right
        bbox_val[2, row_i, col_i] = np.rint((rt - y1)/rt_step_size)  # pixels down
        bbox_val[3, row_i, col_i] = np.rint((mz - x1)/mz_step_size)  # pixels left

    return bbox_val


def save_h5_file(fname, spec_arr, dot_arr, bbox_val,
                 min_rt, max_rt, rt_step_size,
                 min_mz, max_mz, mz_step_size):
    outfile = h5py.File(fname)
    data = outfile.create_dataset("/data",
                                  spec_arr[1:, 1:].shape,
                                  chunks=True,
                                  compression="gzip",
                                  dtype=np.float32,
                                  data=spec_arr[1:, 1:])
    data.attrs['rt_min'] = min_rt
    data.attrs['rt_max'] = max_rt
    data.attrs['rt_step_size'] = rt_step_size
    data.attrs['mz_min'] = min_mz
    data.attrs['mz_max'] = max_mz
    data.attrs['mz_step_size'] = mz_step_size
    
    label = outfile.create_dataset("/label",
                                   dot_arr[1:, 1:].shape,
                                   chunks=True,
                                   compression="gzip",
                                   dtype=np.float32,
                                   data=dot_arr[1:, 1:])
    feature = outfile.create_dataset("/feature",
                                     bbox_val.shape,
                                     chunks=True,
                                     compression="gzip",
                                     dtype=np.float32,
                                     data=bbox_val)
    outfile.close()


def create(in_file="",
           in_featureXML="",
           spec_out="spectrum.png",
           label_out="labels.png",
           feat_out="features.png",
           hdf5_out="data.h5",
           width=1024,
           height=1024,
           rt_step_size=None,
           mz_step_size=None,
           save_png=False):
    """Create .png images for spectra, labels and bounding boxes.

    Spectra are read from mzML file whereas labels and bounding boxes are read
    from featureXML file. Labels image is not generated by OpenMS's
    ImageCreator.

    Arguments:
        in_file: file name of input spectrum file (.mzML)
        in_featureXML: file name of input features file (.featureXML)
        spec_out: name for output spectrum image
        label_out: name for output label image
        feat_out: name for output features image
        hdf5_out: name for HDF5 file
        width: Number of pixels in m/z dimension
                if 0, width defined w.r.t. mz_step_size (default: 1024, min: 0)
        height: Number of pixels in r/t dimension
                if 0, height defined w.r.t rt_step_size (default: 1024, min: 0)
        rt_step_size: step_size for discrete r/t values,
                      used only if width is 0 (default 1)
        mz_step_size: step_size for discrete m/z values,
                      used only if height is 0 (default 0.1)
        save_png: if False, png images will not be saved
    """
    if in_file == "":
        raise AttributeError('Input mzML file not specified')

    if in_featureXML == "":
        raise AttributeError('Input featureXML file not specified.')

    if not spec_out.endswith('.png'):
        raise AttributeError('spec_out should have .png format')

    if not feat_out.endswith('.png'):
        raise AttributeError('feat_out should have .png format')

    if not label_out.endswith('.png'):
        raise AttributeError('label_out should have .png format')

    if not hdf5_out.endswith('.h5'):
        raise AttributeError('hdf5_out should have .h5 format')

    if width < 0:
        raise AttributeError('width should be >= 0')

    if height < 0:
        raise AttributeError('height should be >= 0')

    # load mzML file
    print "Reading", in_file, "..."
    exp = MSExperiment()
    MzMLFile().load(in_file, exp)
    print "Number of Spectra = " + str(exp.size())

    print "Reading", in_featureXML, "..."
    # load featureXML
    feat_map = FeatureMap()
    FeatureXMLFile().load(in_featureXML, feat_map)
    print "number of peptide features = " + str(feat_map.size())

    # create Spectrum Dictionary and store ranges:
    exp.updateRanges(1)
    max_rt, max_mz = exp.getMax()
    min_rt, min_mz = exp.getMin()
    spec_rt_range = (min_rt, max_rt)
    spec_mz_range = (min_mz, max_mz)
    spectrum_dict = _create_spec_dict(exp)

    # create Features Dictionary and store ranges
    feature_dict = _create_feat_dict(feat_map)
    min_rt, min_mz = feat_map.getMin()
    max_rt, max_mz = feat_map.getMax()
    feat_rt_range = (min_rt, max_rt)
    feat_mz_range = (min_mz, max_mz)

    # min_rt = math.floor(min(spec_rt_range[0],
    #                         feat_rt_range[0]))
    # max_rt = math.ceil(max(spec_rt_range[1],
    #                        feat_rt_range[1]))
    # min_mz = math.floor(min(spec_mz_range[0],
    #                         feat_mz_range[0]))
    # max_mz = math.ceil(max(spec_mz_range[1],
    #                        feat_mz_range[1]))
    min_rt, max_rt = spec_rt_range
    min_mz, max_mz = spec_mz_range

    # set height according to rt range
    if height == 0:
        if rt_step_size is None:
            rt_step_size = 1
        height = int((max_rt - min_rt) / rt_step_size)
    else:
        rt_step_size = float(max_rt - min_rt) / height

    # set width according to mz range
    if width == 0:
        if mz_step_size is None:
            mz_step_size = 0.1
        width = int((max_mz - min_mz) / mz_step_size)
    else:
        mz_step_size = float(max_mz - min_mz) / width

    # Extract and grid peak data from MSExperiment
    # Perform Bilinear Interpolation for spectra
    bilip = _interpolation(exp)

    # array for storing specta values using Bilinear Interpolation
    spec_arr = _create_data_array(min_rt, max_rt,
                                  min_mz, max_mz,
                                  height, width)
    spec_arr = _fill_spec_array(spec_arr, bilip, spectrum_dict)
     # flip the array along rows so that the image resembles TOPPView plot
    spec_arr[1:, :] = np.flip(spec_arr[1:, :], 0)
    if save_png is True:
        print "Saving spectra image", spec_out, "..."
        spec_arr = _save_spectrum_image(spec_arr, spec_out)
   
    print "Saving '/data' group in ", hdf5_out, "..."
    outfile = h5py.File(hdf5_out)
    data = outfile.create_dataset("/data",
                                  spec_arr[1:, 1:].shape,
                                  chunks=True,
                                  compression="gzip",
                                  dtype=np.float32,
                                  data=spec_arr[1:, 1:])
    spec_arr = None    # free memory used by spec_arr

    data.attrs['rt_min'] = min_rt
    data.attrs['rt_max'] = max_rt
    data.attrs['rt_step_size'] = rt_step_size
    data.attrs['mz_min'] = min_mz
    data.attrs['mz_max'] = max_mz
    data.attrs['mz_step_size'] = mz_step_size

    # labels array
    dot_arr = _create_data_array(min_rt, max_rt,
                                 min_mz, max_mz,
                                 height, width)
    dot_arr = _insert_dot_values(feature_dict, dot_arr)

    if save_png is True:
        # array for storing feature values for png image
        bbox_arr = np.copy(dot_arr)     # bbox rectangle array contains dot values too    
        bbox_arr = _insert_rect_values(feature_dict, bbox_arr)
        bbox_arr[1:, :] = np.flip(bbox_arr[1:, :], 0)
        print "Saving features image", feat_out, "..."
        bbox_arr = _save_features_image(bbox_arr, feat_out)
        bbox_arr = None

    # array for storing feature as vectors
    bbox_vec = _create_bbox_vectors(feature_dict,
                                    dot_arr,  # bbox vector arr uses dot_arr for index lookup
                                    mz_step_size, 
                                    rt_step_size)

    dot_arr[1:, :] = np.flip(dot_arr[1:, :], 0)
    if save_png is True:
        print "Saving labels image", label_out, "..."
        dot_arr = _save_labels_image(dot_arr, label_out)
    
    print "Saving '/label' group in ", hdf5_out, "..."
    label = outfile.create_dataset("/label",
                                   dot_arr[1:, 1:].shape,
                                   chunks=True,
                                   compression="gzip",
                                   dtype=np.float32,
                                   data=dot_arr[1:, 1:])
    dot_arr = None

    bbox_vec = np.flip(bbox_vec, axis=1)
    print "Saving '/feature' group in", hdf5_out, "..."
    feature = outfile.create_dataset("/feature",
                                     bbox_vec.shape,
                                     chunks=True,
                                     compression="gzip",
                                     dtype=np.float32,
                                     data=bbox_vec)
    bbox_vec = None
    outfile.close()


def _draw_bbox_from_vector(bbox_val):
    # unused, used for debugging bbox vector values

    chan, height, width = bbox_val.shape
    bbox_2 = np.zeros((height, width), dtype=np.int32)

    for i in xrange(height):
        draw_left = True
        draw_right = True
        draw_up = True
        draw_down = True
        rt = i

        for j in xrange(width):
            mz = j
            up = bbox_val[0, i, j]
            right = bbox_val[1, i, j]
            down = bbox_val[2, i, j]
            left = bbox_val[3, i, j]

            if (up + right + down + left) > 0:
                # draw dot
                bbox_2[rt, mz] = 255

                # neighbours of dot
                if rt - 1 > 0:
                    bbox_2[rt - 1, mz] = 255
                if rt + 1 < height:
                    bbox_2[rt + 1, mz] = 255
                if mz - 1 > 0:
                    bbox_2[rt, mz - 1] = 255
                if mz + 1 < width:
                    bbox_2[rt, mz + 1] = 255
                if rt - 1 > 0 and mz - 1 > 0:
                    bbox_2[rt - 1, mz - 1] = 128
                if rt + 1 < height and mz - 1 > 0:
                    bbox_2[rt + 1, mz - 1] = 128
                if rt - 1 > 0 and mz + 1 < width:
                    bbox_2[rt - 1, mz + 1] = 128
                if rt + 1 < height and mz + 1 < width:
                    bbox_2[rt + 1, mz + 1] = 128


                # bounding box
                y1 = mz - left
                y2 = mz + right
                x1 = rt - up
                x2 = rt + down

                if y1 < 0:
                    y1 = 0
                    draw_left = False
                if y2 > width - 1:
                    y2 = width - 1
                    draw_right = False
                if x1 < 0:
                    x1 = 0
                    draw_up = False
                if x2 > height - 1:
                    x2 = height - 1
                    draw_down = False

                if draw_left:
                    for k in range(x1, x2 + 1):
                        bbox_2[k, y1] = 255

                if draw_right:
                    for k in range(x1, x2 + 1):
                        bbox_2[k, y2] = 255

                if draw_up:
                    for k in range(y1, y2 + 1):
                        bbox_2[x1, k] = 255

                if draw_down:
                    for k in range(y1, y2 + 1):
                        bbox_2[x2, k] = 255

            else:
                continue


    rows, cols = bbox_2.shape
    feat_img = Image.new(mode='L',
                        size=(cols, rows),
                        color='white')
    pixels = feat_img.load()
    for i in range(cols):
        for j in range(rows):
            pixels[i, j] = int(255 - bbox_2[j, i])
    #feat_img.save("test_bbox.png")
    feat_img.show()
