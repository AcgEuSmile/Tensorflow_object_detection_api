read -p "please input target directory: " PATH_NAME
if [ -d "./${PATH_NAME}" ]; then
  if [ -d  "./JPEGImages_${PATH_NAME}"  ]; then
    read -p "The output folder is exist, do you want to replace it?(Y/n) " REPLACE
    if [ ${REPLACE} == "Y" ] || [ ${REPLACE} == "y" ]; then
      find ./${PATH_NAME}/ -name "*.jpg"  -type f -print0 | xargs -0 -i cp {} ./JPEGImages_${PATH_NAME}
    else
      echo "Give up replacing"
    fi
  else
    mkdir -p JPEGImages_${PATH_NAME}
    find ./${PATH_NAME}/ -name "*.jpg"  -type f -print0 | xargs -0 -i cp {} ./JPEGImages_${PATH_NAME}
  fi
else
  echo "./${PATH_NAME} directory not found"
fi
