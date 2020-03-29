pipenv run python -m grpc_tools.protoc -I=../share \
  --python_out=. --grpc_python_out=. \
  ../share/services.proto
