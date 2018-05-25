PARENT=/Volumes/MyPassport/merlin/results/cleaned_results
mkdir $PARENT
for SUB in sub-19 sub-20 sub-21 sub-22 sub-23 sub-24 sub-26 sub-27 sub-28 sub-29 sub-30 sub-31 sub-32 sub-33 sub-34 sub-35 sub-36
do
	DATA=/Volumes/MyPassport/merlin/fmriprep/$SUB/${SUB}_task-MerlinMovie_bold_space-MNI152NLin2009cAsym_preproc.nii.gz
	MASK=/Volumes/MyPassport/merlin/fmriprep/$SUB/${SUB}_task-MerlinMovie_bold_space-MNI152NLin2009cAsym_brainmask.nii.gz
	RESULTS_DIR=$PARENT/$SUB
	DETRENDED=$RESULTS_DIR/${SUB}_detrended.nii.gz

	mkdir $RESULTS_DIR
	mkdir $RESULTS_DIR/combined_spaces/
	mkdir $RESULTS_DIR/energy_speech/
	mkdir $RESULTS_DIR/audsemantic_speech/
	mkdir $RESULTS_DIR/visobj_visface/
	mkdir $RESULTS_DIR/audio_visual_semantics/
	mkdir $RESULTS_DIR/visobj_audsem/
	mkdir $RESULTS_DIR/visobj_visnonhumobj/

	python2 trim_and_detrend.py $DATA $DETRENDED

	python2 model_fitting.py $DETRENDED events/audio_energy_events.csv $RESULTS_DIR/audio_energy.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/audio_glove_events.csv $RESULTS_DIR/audio_semantic.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/audio_speech_events.csv $RESULTS_DIR/audio_speech.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/visual_object_events.csv $RESULTS_DIR/visual_object.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/visual_glove_events.csv $RESULTS_DIR/visual_semantic.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/visual_face_events.csv $RESULTS_DIR/visual_face.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/visual_nonhuman_object_events.csv $RESULTS_DIR/visual_nonhuman_object.nii.gz $MASK

	python2 model_fitting.py $DETRENDED events/combined_spaces/energy_speech.csv $RESULTS_DIR/combined_spaces/energy_speech.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/combined_spaces/audglove_speech.csv $RESULTS_DIR/combined_spaces/audsemantic_speech.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/combined_spaces/visobj_visface.csv $RESULTS_DIR/combined_spaces/visobj_visface.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/combined_spaces/audglove_visglove.csv $RESULTS_DIR/combined_spaces/audsemantic_vissemantic.nii.gz $MASK
	python2 model_fitting.py $DETRENDED events/combined_spaces/visobj_audsem_events.csv $RESULTS_DIR/combined_spaces/visobj_audsem_events.nii.gz $MASK

	python2 variance_partitioning.py $RESULTS_DIR/audio_energy.nii.gz $RESULTS_DIR/audio_speech.nii.gz $RESULTS_DIR/combined_spaces/energy_speech.nii.gz $RESULTS_DIR/energy_speech/energy.nii.gz $RESULTS_DIR/energy_speech/speech.nii.gz $RESULTS_DIR/energy_speech/intersection.nii.gz
	python2 variance_partitioning.py $RESULTS_DIR/audio_semantic.nii.gz $RESULTS_DIR/audio_speech.nii.gz $RESULTS_DIR/combined_spaces/audsemantic_speech.nii.gz $RESULTS_DIR/audsemantic_speech/semantic.nii.gz $RESULTS_DIR/audsemantic_speech/speech.nii.gz $RESULTS_DIR/audsemantic_speech/intersection.nii.gz
	python2 variance_partitioning.py $RESULTS_DIR/visual_object.nii.gz $RESULTS_DIR/visual_face.nii.gz $RESULTS_DIR/combined_spaces/visobj_visface.nii.gz $RESULTS_DIR/visobj_visface/object.nii.gz $RESULTS_DIR/visobj_visface/face.nii.gz $RESULTS_DIR/visobj_visface/intersection.nii.gz
	python2 variance_partitioning.py $RESULTS_DIR/visual_semantic.nii.gz $RESULTS_DIR/audio_semantic.nii.gz $RESULTS_DIR/combined_spaces/audsemantic_vissemantic.nii.gz $RESULTS_DIR/audio_visual_semantics/visual.nii.gz $RESULTS_DIR/audio_visual_semantics/audio.nii.gz $RESULTS_DIR/audio_visual_semantics/intersection.nii.gz
	python2 variance_partitioning.py $RESULTS_DIR/visual_object.nii.gz $RESULTS_DIR/audio_semantic.nii.gz $RESULTS_DIR/combined_spaces/visobj_audsem_events.nii.gz $RESULTS_DIR/visobj_audsem/visual.nii.gz $RESULTS_DIR/visobj_audsem/audio.nii.gz $RESULTS_DIR/visobj_audsem/intersection.nii.gz
	python2 variance_partitioning.py $RESULTS_DIR/visual_object.nii.gz $RESULTS_DIR/visual_nonhuman_object.nii.gz $RESULTS_DIR/visual_object.nii.gz $RESULTS_DIR/visobj_visnonhumobj/obj.nii.gz $RESULTS_DIR/visobj_visnonhumobj/nonhuman.nii.gz $RESULTS_DIR/visobj_visnonhumobj/intersection.nii.gz
done

mkdir $PARENT/average
mkdir $PARENT/average/energy_speech/
mkdir $PARENT/average/audsemantic_speech/
mkdir $PARENT/average/visobj_visface/
mkdir $PARENT/average/audio_visual_semantics/
mkdir $PARENT/average/visobj_audsem/
mkdir $PARENT/average/visobj_visnonhumobj/
python3 utils/average_images.py $PARENT/average/audio_energy.nii.gz $PARENT/sub*/audio_energy.nii.gz
python3 utils/average_images.py $PARENT/average/audio_speech.nii.gz $PARENT/sub*/audio_speech.nii.gz
python3 utils/average_images.py $PARENT/average/audio_semantic.nii.gz $PARENT/sub*/audio_semantic.nii.gz
python3 utils/average_images.py $PARENT/average/visual_semantic.nii.gz $PARENT/sub*/visual_semantic.nii.gz
python3 utils/average_images.py $PARENT/average/visual_object.nii.gz $PARENT/sub*/visual_object.nii.gz
python3 utils/average_images.py $PARENT/average/visual_face.nii.gz $PARENT/sub*/visual_face.nii.gz
python3 utils/average_images.py $PARENT/average/visual_nonhuman_object.nii.gz $PARENT/sub*/visual_nonhuman_object.nii.gz

python3 utils/average_images.py $PARENT/average/energy_speech/intersection.nii.gz $PARENT/sub*/energy_speech/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/energy_speech/energy.nii.gz $PARENT/sub*/energy_speech/energy.nii.gz
python3 utils/average_images.py $PARENT/average/energy_speech/speech.nii.gz $PARENT/sub*/energy_speech/speech.nii.gz

python3 utils/average_images.py $PARENT/average/audsemantic_speech/intersection.nii.gz $PARENT/sub*/audsemantic_speech/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/audsemantic_speech/speech.nii.gz $PARENT/sub*/audsemantic_speech/speech.nii.gz
python3 utils/average_images.py $PARENT/average/audsemantic_speech/semantic.nii.gz $PARENT/sub*/audsemantic_speech/semantic.nii.gz

python3 utils/average_images.py $PARENT/average/visobj_visface/intersection.nii.gz $PARENT/sub*/visobj_visface/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_visface/face.nii.gz $PARENT/sub*/visobj_visface/face.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_visface/object.nii.gz $PARENT/sub*/visobj_visface/object.nii.gz

python3 utils/average_images.py $PARENT/average/audio_visual_semantics/intersection.nii.gz $PARENT/sub*/audio_visual_semantics/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/audio_visual_semantics/audio.nii.gz $PARENT/sub*/audio_visual_semantics/audio.nii.gz
python3 utils/average_images.py $PARENT/average/audio_visual_semantics/visual.nii.gz $PARENT/sub*/audio_visual_semantics/visual.nii.gz

python3 utils/average_images.py $PARENT/average/visobj_audsem/intersection.nii.gz $PARENT/sub*/visobj_audsem/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_audsem/visual.nii.gz $PARENT/sub*/visobj_audsem/visual.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_audsem/audio.nii.gz $PARENT/sub*/visobj_audsem/audio.nii.gz

python3 utils/average_images.py $PARENT/average/visobj_visnonhumobj/intersection.nii.gz $PARENT/sub*/visobj_visnonhumobj/intersection.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_visnonhumobj/obj.nii.gz $PARENT/sub*/visobj_visnonhumobj/obj.nii.gz
python3 utils/average_images.py $PARENT/average/visobj_visnonhumobj/nonhuman.nii.gz $PARENT/sub*/visobj_visnonhumobj/nonhuman.nii.gz
