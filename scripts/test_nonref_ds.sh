GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
BOLD_GREEN="\e[1;32m"
ENDSTYLE="\e[0m"

declare -A ds_dict
ds_dict=([U45]="configs/dataset/u45.yaml"
         [RUIE_Color90]="configs/dataset/ruie_color90.yaml"
         [UPoor200]="configs/dataset/upoor200.yaml"
         [UW2023]="configs/dataset/uw2023.yaml")

if [ $# -lt 5 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net_cfg"
    echo -e "3) name"
    echo -e "4) epochs"
    echo -e "5) load_prefix"
    echo -e "for example: \"${BOLD_GREEN}bash ${0} uie configs/network/erd_15blocks_2down.yaml pretrained 299 weights${ENDSTYLE}\""
    exit -1
fi
model_v=${1}
net_cfg=${2}
name=${3}
epoch=${4}
load_prefix=${5}

epochs=$(echo ${raw_epochs} | tr ',' ' ')

for ds_name in ${!ds_dict[@]};
do
    python ./test_${model_v}.py \
    --ds_cfg ${ds_dict[${ds_name}]} \
    --net_cfg ${net_cfg} \
    --name ${name} \
    --test_name ${ds_name} \
    --epoch ${epochs} \
    --load_prefix ${load_prefix}
done