#include <io/dataset_io.h>
#include <io/dataset_io_euroc.h>

namespace visnav {

DatasetIoInterfacePtr DatasetIoFactory::getDatasetIo(
    const std::string &dataset_type) {
  if (dataset_type == "euroc") {
    return DatasetIoInterfacePtr(new EurocIO());
  } else {
    std::cerr << "Dataset type " << dataset_type << " is not supported"
              << std::endl;
    std::abort();
  }
}

}  // namespace visnav