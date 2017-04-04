_dir=data/ATIS_samples
model_dir=model_tmp
max_sequence_length=50  # max length for train/valid/test sequence
task=joint  # available options: intent; tagging; joint
bidirectional_rnn=True  # available options: True; False

python run_multi-task_rnn.py \
      --data_dir $_dir \
      --train_dir   $model_dir\
      --max_sequence_length $max_sequence_length \
      --task $task \
      --bidirectional_rnn $bidirectional_rnn
