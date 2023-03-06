while true; do
    echo -e "Start Grep!\n";
    python grep_data.py;
    python grep_class_data.py;
    echo -e "Sleep 900!\n";
    sleep 900;
done