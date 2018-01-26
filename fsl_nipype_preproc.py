''' Runs the default FSL preprocessing workflow '''

import sys
from nipype import Node
from nipype.interfaces import fsl as fsl
from nipype.interfaces import utility as util
from nipype.interfaces.io import DataSink
from nipype.pipeline import engine as pe
from nipype import LooseVersion


TR = 1.5
highpass_operand = lambda x: '-bptf %.10f -1' % x


def getthreshop(thresh):
    return ['-thr %.10f -Tmin -bin' % (0.1 * val[1]) for val in thresh]


def pickrun(files, whichrun):
    """pick file from list of files"""

    filemap = {'first': 0, 'last': -1, 'middle': len(files) // 2}

    if isinstance(files, list):

        # whichrun is given as integer
        if isinstance(whichrun, int):
            return files[whichrun]
        # whichrun is given as string
        elif isinstance(whichrun, str):
            if whichrun not in filemap.keys():
                raise (KeyError, 'Sorry, whichrun must be either integer index'
                       'or string in form of "first", "last" or "middle')
            else:
                return files[filemap[whichrun]]
    else:
        # in case single file name is given
        return files


def pickvol(filenames, fileidx, which):
    from nibabel import load
    import numpy as np
    from nipype.utils import NUMPY_MMAP
    if which.lower() == 'first':
        idx = 0
    elif which.lower() == 'middle':
        idx = int(
            np.ceil(load(filenames[fileidx], mmap=NUMPY_MMAP).shape[3] / 2))
    elif which.lower() == 'last':
        idx = load(filenames[fileidx]).shape[3] - 1
    else:
        raise Exception('unknown value for volume selection : %s' % which)
    return idx


def getmeanscale(medianvals):
    return ['-mul %.10f' % (10000. / val) for val in medianvals]


def create_preproc(whichvol='middle',
                   whichrun=0):
    version = 0
    if fsl.Info.version() and \
            LooseVersion(fsl.Info.version()) > LooseVersion('5.0.6'):
        version = 507

    """Create a FEAT preprocessing workflow with registration to one volume of the first run
    """

    featpreproc = pe.Workflow(name='preproc')

    """
    Set up a node to define all inputs required for the preprocessing workflow
    """
    inputnode = pe.Node(
        interface=util.IdentityInterface(fields=['func', 'highpass']), name='inputspec')

    """
    Set up a node to define outputs for the preprocessing workflow
    """
    outputnode = pe.Node(
        interface=util.IdentityInterface(fields=[
            'reference', 'motion_parameters', 'realigned_files',
            'motion_plots', 'mask', 'mean', 'highpassed_files'
        ]),
        name='outputspec')

    """
    Convert functional images to float representation. Since there can
    be more than one functional run we use a MapNode to convert each
    run.
    """
    img2float = pe.MapNode(
        interface=fsl.ImageMaths(
            out_data_type='float', op_string='', suffix='_dtype'),
        iterfield=['in_file'],
        name='img2float')
    featpreproc.connect(inputnode, 'func', img2float, 'in_file')

    """
    Extract the middle (or what whichvol points to) volume of the first run as the reference
    """
    extract_ref = pe.Node(
        interface=fsl.ExtractROI(t_size=1),
        iterfield=['in_file'],
        name='extractref')
    featpreproc.connect(img2float, ('out_file', pickrun, whichrun),
                        extract_ref, 'in_file')
    featpreproc.connect(img2float, ('out_file', pickvol, 0, whichvol),
                        extract_ref, 't_min')
    featpreproc.connect(extract_ref, 'roi_file', outputnode, 'reference')

    """
    Realign the functional runs to the reference (`whichvol` volume of first run)
    """
    motion_correct = pe.MapNode(
        interface=fsl.MCFLIRT(
            save_mats=True, save_plots=True, interpolation='spline'),
        name='realign',
        iterfield=['in_file'])
    featpreproc.connect(img2float, 'out_file', motion_correct, 'in_file')
    featpreproc.connect(extract_ref, 'roi_file', motion_correct, 'ref_file')

    featpreproc.connect(motion_correct, 'par_file', outputnode,
                        'motion_parameters')
    featpreproc.connect(motion_correct, 'out_file', outputnode,
                        'realigned_files')
    """
    Plot the estimated motion parameters
    """
    plot_motion = pe.MapNode(
        interface=fsl.PlotMotionParams(in_source='fsl'),
        name='plot_motion',
        iterfield=['in_file'])
    plot_motion.iterables = ('plot_type', ['rotations', 'translations'])
    featpreproc.connect(motion_correct, 'par_file', plot_motion, 'in_file')
    featpreproc.connect(plot_motion, 'out_file', outputnode, 'motion_plots')

    """
    Extract the mean volume of the first functional run
    """
    meanfunc = pe.Node(
        interface=fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
        name='meanfunc')
    featpreproc.connect(motion_correct, ('out_file', pickrun, whichrun),
                        meanfunc, 'in_file')

    """
    Strip the skull from the mean functional to generate a mask
    """
    meanfuncmask = pe.Node(
        interface=fsl.BET(mask=True, no_output=True, frac=0.3),
        name='meanfuncmask')
    featpreproc.connect(meanfunc, 'out_file', meanfuncmask, 'in_file')

    """
    Mask the functional runs with the extracted mask
    """
    maskfunc = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_bet', op_string='-mas'),
        iterfield=['in_file'],
        name='maskfunc')
    featpreproc.connect(motion_correct, 'out_file', maskfunc, 'in_file')
    featpreproc.connect(meanfuncmask, 'mask_file', maskfunc, 'in_file2')

    """
    Determine the 2nd and 98th percentile intensities of each functional run
    """
    getthresh = pe.MapNode(
        interface=fsl.ImageStats(op_string='-p 2 -p 98'),
        iterfield=['in_file'],
        name='getthreshold')
    featpreproc.connect(maskfunc, 'out_file', getthresh, 'in_file')

    """
    Threshold the first run of the functional data at 10% of the 98th percentile
    """
    threshold = pe.MapNode(
        interface=fsl.ImageMaths(out_data_type='char', suffix='_thresh'),
        iterfield=['in_file', 'op_string'],
        name='threshold')
    featpreproc.connect(maskfunc, 'out_file', threshold, 'in_file')

    """
    Define a function to get 10% of the intensity
    """
    featpreproc.connect(getthresh, ('out_stat', getthreshop), threshold,
                        'op_string')

    """
    Determine the median value of the functional runs using the mask
    """
    medianval = pe.MapNode(
        interface=fsl.ImageStats(op_string='-k %s -p 50'),
        iterfield=['in_file', 'mask_file'],
        name='medianval')
    featpreproc.connect(motion_correct, 'out_file', medianval, 'in_file')
    featpreproc.connect(threshold, 'out_file', medianval, 'mask_file')

    """
    Dilate the mask
    """
    dilatemask = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_dil', op_string='-dilF'),
        iterfield=['in_file'],
        name='dilatemask')
    featpreproc.connect(threshold, 'out_file', dilatemask, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', outputnode, 'mask')

    """
    Mask the motion corrected functional runs with the dilated mask
    """
    maskfunc2 = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_mask', op_string='-mas'),
        iterfield=['in_file', 'in_file2'],
        name='maskfunc2')
    featpreproc.connect(motion_correct, 'out_file', maskfunc2, 'in_file')
    featpreproc.connect(dilatemask, 'out_file', maskfunc2, 'in_file2')

    """
    Scale the median value of the run is set to 10000
    """
    meanscale = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_gms'),
        iterfield=['in_file', 'op_string'],
        name='meanscale')
    featpreproc.connect(maskfunc2, 'out_file', meanscale, 'in_file')

    """
    Define a function to get the scaling factor for intensity normalization
    """
    featpreproc.connect(medianval, ('out_stat', getmeanscale), meanscale,
                        'op_string')

    """
    Generate a mean functional image from the first run
    """
    meanfunc3 = pe.Node(
        interface=fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
        iterfield=['in_file'],
        name='meanfunc3')

    """
    Do highpass temporal filtering
    """
    featpreproc.connect(meanscale, ('out_file', pickrun, whichrun), meanfunc3,
                        'in_file')
    featpreproc.connect(meanfunc3, 'out_file', outputnode, 'mean')

    highpass = pe.MapNode(
        interface=fsl.ImageMaths(suffix='_tempfilt'),
        iterfield=['in_file'],
        name='highpass')
    featpreproc.connect(inputnode, ('highpass', highpass_operand),
                        highpass, 'op_string')
    featpreproc.connect(meanscale, 'out_file', highpass, 'in_file')

    if version < 507:
        featpreproc.connect(highpass, 'out_file', outputnode,
                            'highpassed_files')
    else:
        """
        Add back the mean removed by the highpass filter operation as of FSL 5.0.7
        """
        meanfunc4 = pe.MapNode(
            interface=fsl.ImageMaths(op_string='-Tmean', suffix='_mean'),
            iterfield=['in_file'],
            name='meanfunc4')

        featpreproc.connect(meanscale, 'out_file', meanfunc4, 'in_file')
        addmean = pe.MapNode(
            interface=fsl.BinaryMaths(operation='add'),
            iterfield=['in_file', 'operand_file'],
            name='addmean')
        featpreproc.connect(highpass, 'out_file', addmean, 'in_file')
        featpreproc.connect(meanfunc4, 'out_file', addmean, 'operand_file')
        featpreproc.connect(addmean, 'out_file', outputnode,
                            'highpassed_files')

    return featpreproc


if __name__ == '__main__':
    sub_id = sys.argv[1]
    preproc = create_preproc()
    preproc.inputs.inputspec.func = ['/Volumes/MyPassport/merlin/data/%s/func/%s_task-MerlinMovie_bold.nii.gz' % (sub_id, sub_id)]
    preproc.inputs.inputspec.highpass = 120 / (2.0 * TR)
    preproc.base_dir = '/Volumes/MyPassport/merlin/dump/%s' % sub_id

    datasink = Node(DataSink(), name='datasink')
    datasink.inputs.base_directory = '/Volumes/MyPassport/merlin/preprocessed_data/%s/' % sub_id
    preproc.connect([(preproc.get_node('outputspec'), datasink, [('motion_parameters', 'motion'),
                                                                 ('motion_plots', 'motion_plots'),
                                                                 ('mask', 'mask'),
                                                                 ('realigned_files', 'realigned_files'),
                                                                 ('mean', 'mean'),
                                                                 ('reference', 'reference'),
                                                                 ('highpassed_files', 'highpassed_files')])])
    preproc.run()
