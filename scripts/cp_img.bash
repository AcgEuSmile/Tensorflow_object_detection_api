read -p "Please input target directory: " PATH_NAME
if [ -d "./${PATH_NAME}" ]; then
  if [ ! -d  "./JPEGImages_${PATH_NAME}"  ]; then
    mkdir -p JPEGImages_${PATH_NAME}
    find ./${PATH_NAME}/ -name "*.jpg"  -type f -print0 | xargs -0 -i cp {} ./JPEGImages_${PATH_NAME}
  fi
else
  echo "./${PATH_NAME} directory not found"
fi