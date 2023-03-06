while true; do
    echo -e "Start Stage 1!\n";
    python run.py --stage 1 --gpus 0,1,2,3,4,5,6 --thread 3;
    sleep 10800;
    echo -e "Stop commands!\n";
    bash commands/stop_commands1.sh;
    sleep 300;
done