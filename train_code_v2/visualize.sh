train_data="../metadata/train_fc7"
train_label="../metadata/train_label"
test_data="../metadata/test_fc7"
test_label="../metadata/test_label"
all_info="../metadata/all_entity"
word_vector="../metadata/word_vector"
hidden_size=50
projection_size=10
reg_lambda=0.01
learning_rate=0.01
improvement_rate=0.05
max_epoch=1000
save_model="../model/pro"$projection_size".h"$hidden_size
#load_model=


# train
#python train.py --visualize $train_data $train_label --all-info $all_info --word-vector $word_vector --hidden-size $hidden_size --proj-size $projection_size --reg-lambda $reg_lambda --learning-rate $learning_rate --max-epoch $max_epoch --save-net $save_model 

# test
python train.py --visualize $train_data $train_label --all-info $all_info --word-vector $word_vector --hidden-size $hidden_size --proj-size $projection_size --reg-lambda $reg_lambda --learning-rate $learning_rate --max-epoch $max_epoch --load-net $save_model 
