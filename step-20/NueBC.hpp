#ifndef _H_NUEBC__
#define _H_NUEBC__

#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/point.h>
#include <vector>

namespace elas{

template< int dim>
class NueBC
{
public:
  NueBC (){};

  virtual
  dealii::SymmetricTensor<2,dim>
  value(const dealii::Point<dim> &p) const;
  
  virtual
  void
  value_list (const std::vector<dealii::Point<dim> > & points,
	      std::vector<dealii::SymmetricTensor<2,dim> >    & values) const;  
};

} //end Namespace elas

#endif
