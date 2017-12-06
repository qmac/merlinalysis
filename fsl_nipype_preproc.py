''' Runs the default FSL preprocessing workflow '''

from os.path import join
from nipype import Node
from nipype.interfaces.io import DataSink
from nipype.workflows.fmri.fsl import create_featreg_preproc


data_dir = '/Users/quinnmac/Documents/NeuralComputation/project/data/sub-21'
TR = 1.5

preproc = create_featreg_preproc()
preproc.inputs.inputspec.func = [join(data_dir, 'sub-21_task-MerlinMovie_bold.nii.gz')]
preproc.inputs.inputspec.fwhm = 5
preproc.inputs.inputspec.highpass = 120 / (2.0 * TR)
preproc.base_dir = data_dir

datasink = Node(DataSink(), name='datasink')
datasink.inputs.base_directory = data_dir
preproc.connect([(preproc.get_node('outputspec'), datasink, [('motion_parameters', 'motion'),
                                                             ('motion_plots', 'motion_plots'),
                                                             ('mask', 'mask'),
                                                             ('realigned_files', 'realigned_files'),
                                                             ('smoothed_files', 'smoothed_files'),
                                                             ('mean', 'mean'),
                                                             ('reference', 'reference'),
                                                             ('highpassed_files', 'highpassed_files')])])
preproc.run()
