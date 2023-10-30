GREEN="\e[32m"
RED="\e[31m"
BOLD="\e[1m"
ENDSTYLE="\e[0m"

declare -A ds_dict
ds_dict=([U45]="configs/dataset/u45.yaml"
         [RUIE_Color90]="configs/dataset/ruie_color90.yaml"
         [UPoor200]="configs/dataset/upoor200.yaml"
         [UW2023]="configs/dataset/uw2023.yaml")

if [ $# -lt 4 ]
then
    echo -e "${RED}PLEASE PASS IN THE FOLLOWING ARGUMENTS IN ORDER!${ENDSTYLE}"
    echo -e "1) model_v"
    echo -e "2) net_cfg"
    echo -e "3) name"
    echo -e "4) epoch"
    echo -e "for example: ${BOLD}bash ${0} ie configs/network/ra_9blocks_2down.yaml LSUI_01 299${ENDSTYLE}"
    exit -1
fi
model_v=${1}
net_cfg=${2}
name=${3}
epoch=${4}

for ds_name in ${!ds_dict[@]};
do
    python ./test_${model_v}.py \
    --ds_cfg ${ds_dict[${ds_name}]} \
    --net_cfg ${net_cfg} \
    --name ${name} \
    --test_name ${ds_name} \
    --epoch ${epoch}
done