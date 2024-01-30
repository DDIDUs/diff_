for model in vggnet resnet densenet pyramidnet custom
do
    python3 main.py \
        -dataset cifar10 \
        -train_model $model \
        -repeat_num 3 \
        &> train_$model.log
done
