#ifndef _H_ELASRHS__
#define _H_ELASRHS__

#include "globals.hpp"

namespace elas {

template <int dim>
class elasRHS : public dealii::TensorFunction<1,dim>
{
public:
  elasRHS () : dealii::TensorFunction<1,dim>() {}

  virtual dealii::Tensor<1,dim> value (const dealii::Point<dim> &p) const;

  virtual void value_list (const std::vector<dealii::Point<dim> > &points,
			   std::vector<dealii::Tensor<1,dim> > &values) const;

};

}//end namespace elas

#endif
