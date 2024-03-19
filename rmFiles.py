import os
from glob import glob
def rmFiles(root):
    #Remove image files
    images = glob(root+'/*/**/'+'image*',recursive=True)
    for image in images:
        os.remove(image)
    images = glob(root+'/*/**/'+'result*',recursive=True)
    for image in images:
        os.remove(image)

root = 'E:/prostateIntrafractionMotion/masks'
rmFiles(root)