#!/bin/bash
echo "Start training task"

while getopts ":b:" opt; do
  case $opt in
    b)
      case $OPTARG in
        sgan)
          cd ./models/sgan/scripts/
          python train_DADA.py
          ;;
        trajectron++)
          cd ./models/trajectron++/
          python train_DADA.py
          ;;
        tutr)
          cd ./models/tutr/
          python train_DADA.py
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