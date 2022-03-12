import numpy as np
import onnxruntime as rt
import os
import cv2
import time
from gooey import Events
from gooey import Gooey, GooeyParser
from gooey import types as t
import sys
from pathvalidate import ValidationError, validate_filepath
import glob 
from shutil import rmtree
# import subprocess # for checking GPU presence

################################## PyInstaller  Utility ##################################
def deleteOldPyinstallerFolders(time_threshold = 3600): 
    """
    PyInstaller creates temporary folder at every run of executable program and fails 
    to delete it if the program is not closed correctly (this includes using the close button 
    on windows title bar). All such folders have a prefix '_MEI'. So we delete such folders 
    if they are older than time_threshold.

    :param int time_threshold: _MEI files modified before time_threshold (in sec) from 
                                current time will be deleted.
    """
    # Default setting: Remove after 1 hour, time_threshold in seconds
    try:
        base_path = sys._MEIPASS
    except Exception:
        return 

    temp_path = os.path.abspath(os.path.join(base_path, '..')) # Go to parent folder of MEIPASS

    # Search all MEIPASS folders and delete them 
    mei_folders = glob.glob(os.path.join(temp_path, '_MEI*'))
    for item in mei_folders:
        if (time.time()-os.path.getctime(item)) > time_threshold:
            rmtree(item)

deleteOldPyinstallerFolders() # delelte old temporary folders

def resource_path(relative_path):
    """
    PyInstaller creates a temporary folder at every run of executable program and unpacks resources. 
    We need to use resources from this temperory folder. Using relative path (to python script) of resources 
    will result in error.

    :param str relative_path: Path to a resource
    :return: Path of the resource in the temporary folder
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)

########################################################## Gooey Argument Validators ##########################################################
def input_file_validator(value):
    """ 
    Input should be a valid image file. Supported extensions: ['.jpg','.jpeg','.png','.bmp','.webp'] 
    """
    if os.path.isfile(value) and os.path.splitext(os.path.basename(value))[1] in ['.jpg','.jpeg','.png','.bmp','.webp']:
        return value
    else: 
        raise TypeError("Select a valid input file.")

def batch_input_file_validator(value):
    """ 
    All input files should be valid image files. Supported extensions: ['.jpg','.jpeg','.png','.bmp','.webp'] 
    """
    for i in value:
        if not os.path.isfile(value):
            raise TypeError("Select a valid input file.")
    return value

def input_dir_validator(value):
    """ 
    Should be a valid directory. 
    """
    if os.path.isdir(value):
        return value
    else:
        raise TypeError("Select a valid directory.")

def output_dir_validator(value):
    dir_name = os.path.dirname(value)
    if dir_name == '':
        return './' 
    if os.path.isdir(dir_name):
        return value
    else:
        try:
            os.makedirs(os.path.dirname(value))
        except OSError:
            raise TypeError("Select a valid directory.")

def output_file_validator(value):
    """ 
    1. If directory does not exist, try creating it. Fails if valid directory cannot be created.
    2. File name should be valid. 
    """
    if value is None:
        return value
    else:
        try:
            validate_filepath(os.path.basename(value))
        except ValidationError as e:
            raise TypeError("Select a valid filename.")
        
        try:
            dir_name = os.path.dirname(value)
            if dir_name == '':
                return value 
            if os.path.isdir(dir_name):
                return value
            else:
                try:
                    os.makedirs(os.path.dirname(value))
                except OSError:
                    raise TypeError("Select a valid filename.")
        except:
            raise TypeError("Select a valid filename.")

def window_validator(value):
    """ 
    Supported window values: 4, 8, 16
    """
    if not value.isnumeric:
        raise TypeError("Select a valid vlaue.")
    elif int(value) in (4, 8, 16):
        return int(value)
    else:
        raise TypeError("Select a valid vlaue.")

def tile_validator(value):
    """ 
    Supported tile values: 128, 256, 320, 448, 512
    """
    if not value.isnumeric:
        raise TypeError("Select a valid vlaue.")
    elif value.isnumeric and int(value) in (128, 256, 320, 448, 512):
        return int(value)
    else:
        raise TypeError("Select a valid vlaue.")

def overlap_validator(value):
    """ 
    Supported tile overlap values: 32, 64
    """
    if not value.isnumeric:
        raise TypeError("Select a valid vlaue.")
        return
    elif int(value) in (32, 64):
        return int(value)
    else:
        raise TypeError("Select a valid vlaue.")

def border_validator(value):
    """ 
    Supported border values: 0
    """
    if not value.isnumeric:
        raise TypeError("Select a valid vlaue.")
    elif int(value) == 0:
        return int(value)
    else:
        raise TypeError("Select a valid vlaue.")

########################################################## Gooey App ##########################################################
@Gooey(program_name="4x Image Enlarger", 
        show_success_modal=False, 
        default_size=(700, 680),
        use_events=[Events.VALIDATE_FORM], 
        navigation='TABBED',
        dump_build_config = False,
        show_restart_button=False, # THIS WON'T WORK (REMINDER: I've disabled restart button directly in the conda package.)
        image_dir=resource_path('images'),
        menu=[{
        'name': 'File',
        'items': [{
                'type': 'AboutDialog',
                'menuTitle': 'About',
                'name': '4x Image Enlarger',
                'description': 'Enlarge any image by 4x',
                'version': '0.1',
                'copyright': '2022',
                'developer': 'scholarstree',
                'license': 'MIT'
            }]
        },{
        'name': 'Help',
        'items': [{
            'type': 'Link',
            'menuTitle': 'Documentation',
            'url': '-'
        }]
    }])
def get_args():
    parser = GooeyParser(description="v0.1")

    # Subparsers for TABBED view. 3 tabs: File Enlarge, Batch Enlarge, Directory Enlarge
    subparsers = parser.add_subparsers(help="commands", dest="command") 

    ################################## File Enlarge ##################################
    file = subparsers.add_parser("file_enlarge", prog="File Enlarge")
    file_group1 = file.add_argument_group('Basic Settings')
    file_group2 = file.add_argument_group('Advanced Settings (optional)')

    file_group1.add_argument('-i', '--input_file', metavar="Input File", type=input_file_validator,
                        widget='FileChooser', 
                        gooey_options={'wildcard':"Image files (jpg, jpeg, png, bmp, webp)|*.jpg;*.jpeg;*.png;*bmp;*webp|""All files (*.*)|*.*",
                                        'default_dir': "./",
                                        'message': "Select input file"},
                        required=True,
                        help="Image to enlarge")                   
    file_group1.add_argument('-o', '--output_file', metavar="Output File", type=output_file_validator,
                        widget='FileSaver', 
                        gooey_options={'wildcard':"Image files jpg,jpeg,png,bmp,webp|*.jpg;*.jpeg;*.png;*bmp;*webp|""All files (*.*)|*.*",
                                        'default_dir': "./", 
                                        'message': "Select output filename"}, 
                        default="./enlarged_images/image_enlarged_4x.png",
                        help="Output filename")
    file_group1.add_argument('-p', '--patch_inference', metavar="Patch Inference", widget='CheckBox', action="store_true", help="Patch inference if using large input resolution")
    file_group2.add_argument('-w', '--window_size', metavar="Window Size", default=8, help="Window size | Valid values: 4, 8, 16", type=window_validator)
    file_group2.add_argument('-b', '--border', metavar="Border", default=0, help="Border | Valid values: 0", type=border_validator)
    file_group2.add_argument('-t', '--tile', metavar="Tile", default=256, help="Patch inference | Valid values: 128, 256, 320, 448, 512", type=tile_validator)
    file_group2.add_argument('-to', '--tile_overlap', metavar="Tile Overlap", default=32, help="Tile Overlap | Valid values: 32, 64", type=overlap_validator)

    ################################## Batch Enlarge ##################################
    batch = subparsers.add_parser("batch_enlarge", prog="Batch Enlarge")
    batch_group1 = batch.add_argument_group('Basic Settings')
    batch_group2 = batch.add_argument_group('Advanced Settings (optional)')

    batch_group1.add_argument('-bi', '--input_batch', metavar="Batch Input Files", type=batch_input_file_validator,
                        widget='MultiFileChooser', nargs='+',
                        gooey_options={'wildcard':"Image files (jpg, jpeg, png, bmp, webp)|*.jpg;*.jpeg;*.png;*bmp;*webp|"
                                                    "All files (*.*)|*.*",
                                        'default_dir': "./images",
                                        'message': "Select input file"},
                        required=True,
                        help="Images to enlarge")                   
    batch_group1.add_argument('-bo', '--output_batch', metavar="Output Directory", type=output_dir_validator,
                        widget='DirChooser', 
                        default="./enlarged_images",
                        help="Directory for enlarged files")
    batch_group1.add_argument('-bp', '--bpatch_inference', metavar="Patch Inference", widget='CheckBox', action="store_true", help="Patch inference if using large input resolution")
    batch_group2.add_argument('-bw', '--bwindow_size', metavar="Window Size", default=8, help="Window size | Valid values: 8, 16", type=window_validator)
    batch_group2.add_argument('-bb', '--bborder', metavar="Border", default=0, help="Border | Valid values: 0", type=border_validator)
    batch_group2.add_argument('-bt', '--btile', metavar="Tile", default=256, help="Patch inference | Valid values: 128, 256, 320, 448, 512", type=tile_validator)
    batch_group2.add_argument('-bto', '--btile_overlap', metavar="Tile Overlap", default=32, help="Tile Overlap | Valid values: 32, 64", type=overlap_validator)

    ################################## Directory Enlarge ##################################
    dir = subparsers.add_parser("dir_enlarge", prog="Directory Enlarge")
    dir_group1 = dir.add_argument_group('Basic Settings')
    dir_group2 = dir.add_argument_group('Advanced Settings (optional)')

    dir_group1.add_argument('-di', '--input_dir', metavar="Input Directory", type=input_dir_validator,
                        widget='DirChooser', 
                        gooey_options={'default_dir': "./images",
                                        'message': "Select input file"},
                        required=True,
                        help="Directory in which images to enlarge")                   
    dir_group1.add_argument('-do', '--output_dir', metavar="Output Directory", type=output_dir_validator,
                        widget='DirChooser', 
                        default="./enlarged_images",
                        help="Directory for enlarged files")
    dir_group1.add_argument('-dp', '--dpatch_inference', metavar="Patch Inference", widget='CheckBox', action="store_true", help="Patch inference if using large input resolution")
    dir_group2.add_argument('-dw', '--dwindow_size', metavar="Window Size", default=8, help="Window size | Valid values: 8, 16", type=window_validator)
    dir_group2.add_argument('-db', '--dborder', metavar="Border", default=0, help="Border | Valid values: 0", type=border_validator)
    dir_group2.add_argument('-dt', '--dtile', metavar="Tile", default=256, help="Patch inference | Valid values: 128, 256, 320, 448, 512", type=tile_validator)
    dir_group2.add_argument('-dto', '--dtile_overlap', metavar="Tile Overlap", default=32, help="Tile Overlap | Valid values: 32, 64", type=overlap_validator)

    return parser.parse_args()

########################################################## Utility Functions ##########################################################
def time_format(seconds):
    """
    Convert seconds into DATE HOUR MIN SEC format.
    """
    if seconds is not None:
        seconds = int(seconds)
        d = seconds // (3600 * 24)
        h = seconds // 3600 % 24
        m = seconds % 3600 // 60
        s = seconds % 3600 % 60
        if d > 0:
            return '{:02d}D {:02d}H {:02d}m {:02d}s'.format(d, h, m, s)
        elif h > 0:
            return '{:02d}H {:02d}m {:02d}s'.format(h, m, s)
        elif m > 0:
            return '{:02d}m {:02d}s'.format(m, s)
        elif s > 0:
            return '{:02d}s'.format(s)
    return '-'

def inference(input_img_path, sess, scale, window_size, tile, tile_overlap, output_img_path, patch_inference, args):
    """
    Loads image and returns enlarged image. Can be called in loop for multiple images.
    """
    (img_name, img_ext) = os.path.splitext(os.path.basename(input_img_path))
    img_old = cv2.imread(input_img_path, cv2.IMREAD_COLOR).astype(np.float32) / 255.
    img_old = np.transpose(img_old if img_old.shape[2] == 1 else img_old[:, :, [2, 1, 0]], (2, 0, 1))  # (width, height, channel) to (channel, width, height)
    img_old = np.expand_dims(img_old, axis=0)

    _, _, h_old, w_old = img_old.shape
    print("Loaded image: {} | Resolution: ({} x {})".format(input_img_path, w_old, h_old))
    sys.stdout.flush()

    h_pad = (h_old // window_size + 1) * window_size - h_old
    w_pad = (w_old // window_size + 1) * window_size - w_old
    img_old = np.concatenate([img_old, np.flip(np.copy(img_old), [2])], 2)[:, :, :h_old + h_pad, :]
    img_old = np.concatenate([img_old, np.flip(np.copy(img_old), [3])], 3)[:, :, :, :w_old + w_pad]

    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    if (h_old * w_old > 250000 and patch_inference is True):
        print("WARNING: Image size is large. Try turning on Patch Inference if this run fails.")

    print("Enlarging...")
    sys.stdout.flush()

    start_e = time.time()
    if patch_inference is False: # inference for whole image
        output = sess.run([label_name], {input_name: img_old})[0]
    else: # inference for image in patches
        b, c, h, w = img_old.shape
        tile = min(tile, h, w)
        assert tile % window_size == 0, "tile size should be a multiple of window_size"
        sf = scale

        stride = tile - tile_overlap
        h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
        w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
        E = np.zeros((b, c, h*sf, w*sf), dtype=np.float32)
        W = np.zeros_like(E)

        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_old[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch = sess.run([label_name], {input_name: in_patch})[0]
                out_patch_mask = np.ones_like(out_patch)

                E[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch
                W[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf] += out_patch_mask
        output = E / (W+0.00001)
    end_e = time.time()

    print("Enlarged. Time taken: ", time_format(end_e - start_e))
    print("Post Processing...")
    sys.stdout.flush()
    
    output = np.squeeze(output)
    output = np.clip(output, 0., 1.) 
    if output.ndim == 3:
        output = np.transpose(output[[2, 1, 0], :, :], (1, 2, 0)) # (channel, width, height) to (width, height, channel)

    output = (output * 255.0).round().astype(np.uint8)  # float32 to uint8
    (w_new, h_new, _) = output.shape
    cv2.imwrite(output_img_path, output) # save image

    print("DONE! Enlarged file saved as {} | Resolution: ({} x {})".format(output_img_path, w_new, h_new))
    print("---------------------------------")
    sys.stdout.flush()

def enlarge(args):
    """
    Loads onnx file once and loops over input images for enlargement.
    """
    print("Using CPU inference. Loading model...")
    start_model = time.time()
    sys.stdout.flush()
    sess = rt.InferenceSession(resource_path('./models/model-4x.onnx'))

    ################################## GPU Inference ##################################

    # ONNX GPU inference depends on CUDA version. Hence, not used for executable program.

    # try:
    #     subprocess.check_output('nvidia-smi')
    #     print('Nvidia GPU detected! Using GPU for inference.')
    #     sess = rt.InferenceSession("model-4x.onnx", None, providers=["CUDAExecutionProvider"])
    # except Exception:
    #     print('No Nvidia GPU in system. Using CPU for inference.')
    #     sess = rt.InferenceSession(resource_path('model-4x.onnx'), None, providers=["CPUExecutionProvider"])

     ###################################################################################
    end_model = time.time()
    print("Model loaded. Time taken: ", time_format(end_model - start_model))
    sys.stdout.flush()
    
    scale = 4

    # Extract File Enlarge arguments and call inference on the input image.
    if args.command == "file_enlarge":
        print("\n--------------------------------- File Enlarge Started ---------------------------------")
        tile = args.tile
        tile_overlap = args.tile_overlap
        border = args.border
        window_size = args.window_size
        (img_name, img_ext) = os.path.splitext(os.path.basename(args.output_file))
        enlarged_img_ext = '.png'
        save_dirname = os.path.dirname(args.output_file) if os.path.dirname(args.output_file) is not None else os.path.dirname(args.input_file)
        file_output = os.path.join(save_dirname, img_name + enlarged_img_ext)
        inference(args.input_file, sess, scale, window_size, tile, tile_overlap, file_output, args.patch_inference, args)
        print("--------------------------------- File Enlarge Complete ---------------------------------\n\n")

    # Extract Batch Enlarge arguments and call inference on all the input images.
    if args.command == "batch_enlarge":
        print("\n--------------------------------- Batch Enlarge Started ---------------------------------")
        tile = args.btile
        tile_overlap = args.btile_overlap
        border = args.bborder
        window_size = args.bwindow_size

        print("{} images to enlarge.".format(len(args.input_batch)))

        for i, input_img_path in enumerate(args.input_batch):
            print("\n---------------------------------")
            print("Enlarging {} of {} images.".format(i+1, len(args.input_batch)))
            (img_name, img_ext) = os.path.splitext(os.path.basename(input_img_path))
            enlarged_img_ext = '.png'
            file_output = os.path.join(args.output_batch, "enlarged4x_" + img_name + enlarged_img_ext)
            inference(input_img_path, sess, scale, window_size, tile, tile_overlap, file_output, args.bpatch_inference, args)
        print("--------------------------------- Batch Enlarge Complete ---------------------------------\n\n")

    # Extract Directory Enlarge arguments and call inference on all the images in the directory.
    if args.command == "dir_enlarge":
        print("\n--------------------------------- Directory Enlarge Started ---------------------------------")
        tile = args.dtile
        tile_overlap = args.dtile_overlap
        border = args.dborder
        window_size = args.dwindow_size

        input_files_list = []
        valid_image_types = ['*.jpg','*.jpeg','*.png','*.bmp','*.webp'] 
        for img in valid_image_types:
            input_files_list.extend(glob.glob(os.path.join(args.input_dir, img)))

        print("{} images to enlarge.".format(len(input_files_list)))

        for i, input_img_path in enumerate(input_files_list):
            print("\n---------------------------------")
            print("Enlarging {} of {} images.".format(i+1, len(input_files_list)))
            (img_name, img_ext) = os.path.splitext(os.path.basename(input_img_path))
            enlarged_img_ext = '.png'
            file_output = os.path.join(args.output_dir, "enlarged4x_" + img_name + enlarged_img_ext)
            inference(input_img_path, sess, scale, window_size, tile, tile_overlap, file_output, args.dpatch_inference, args)
        print("--------------------------------- Directory Enlarge Complete ---------------------------------\n\n")

########################################################## Main Function ##########################################################
def main():
    args = get_args()

    enlarge(args)

########################################################## Run Script ##########################################################
if __name__ == '__main__':
    main()