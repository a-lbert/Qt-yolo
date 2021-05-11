# import pcl
# import numpy as np
# p = pcl.PointCloud(np.array([[1, 2, 3], [3, 4, 5]], dtype=np.float32))
# seg = p.make_segmenter()
# # seg.set_model_type(pcl.SACMODEL_PLANE)
# # seg.set_method_type(pcl.SAC_RANSAC)
# # indices, model = seg.segment()

import sys
import pydoc

import sys
import pydoc

def output_help_to_file(filepath, request):
    f = open(filepath, 'w')
    sys.stdout = f
    pydoc.help(request)
    f.close()
    sys.stdout = sys.__stdout__
    return

output_help_to_file(r'/home/limeng/Documents/_pcl.txt', 'pcl._pcl')