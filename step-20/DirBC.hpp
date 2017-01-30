#ifndef _H_DIRBC__
#define _H_DIRBC__

#include "globals.hpp"

namespace elas{

template <int dim>
class DirBC : public dealii::TensorFunction<1,dim>
{
public:
  DirBC () : dealii::TensorFunction<1,dim>() {}

  virtual dealii::Tensor<1,dim> value (const dealii::Point<dim> &p) const;

  virtual void value_list (const std::vector<dealii::Point<dim> > &points,
			   std::vector<dealii::Tensor<1,dim> > & values) const;
};

} // end namespace elas
#endif
