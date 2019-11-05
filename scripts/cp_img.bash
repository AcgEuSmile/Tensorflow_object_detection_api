# read input
read -p "Please input target directory: " PATH_NAME
# if "./${PATH_NAME}" exist, then 
if [ -d "./${PATH_NAME}" ]; then
  # if "./JPEGImages_${PATH_NAME}" doesn't exit, then
  if [ ! -d  "./JPEGImages_${PATH_NAME}"  ]; then
    # make directory
    mkdir -p JPEGImages_${PATH_NAME}
    # find all jpg "files" and copy to "./JPEGImages_${PATH_NAME}"
    find ./${PATH_NAME}/ -name "*.jpg"  -type f -print0 | xargs -0 -i cp {} ./JPEGImages_${PATH_NAME}
  fi
else
  # if "./${PATH_NAME}" exist, then show error message.
  echo "./${PATH_NAME} directory not found"
fi
