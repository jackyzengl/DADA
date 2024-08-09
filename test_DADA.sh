#!/bin/bash
echo "Start testing task"

while getopts ":b:" opt; do
  case $opt in
    b)
      case $OPTARG in
        sgan)
          cd ./models/sgan/scripts/
          python evaluate_model.py
          ;;
        trajectron++)
          cd ./models/trajectron++/
          python test.py
          ;;
        tutr)
          cd ./models/tutr/
          python evaluate_model.py
          ;;
        *)
          echo "Invalid argument: -b $OPTARG"
          exit 1
          ;;
      esac
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument."
      exit 1
      ;;
  esac
done
echo "Done."