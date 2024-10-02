echo "Run PINN experiment"

dcgd_type_list=('avg','proj','center')
equation_list=('helmholtz','burgers','klein-gordon')
lr_list=(1e-3,1e-4,1e-5)

for dcgd_type in ${dcgd_type_list[@]}
do 
    for equation in ${equation_list[@]}
    do
        for lr in ${lr_list[@]}
        do
            python main.py --equation=$equation --dcgd=$dcgd_type --lr=$lr --optim='adam' --depth=3 --width=50 --batch=128
        done
    done
done

echo "Finish"