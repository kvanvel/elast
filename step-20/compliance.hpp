#ifndef H_COMPLIANCE__
#define H_COMPLIANCE__

#include <vector>
#include <deal.II/base/symmetric_tensor.h>
#include <deal.II/base/point.h>

namespace elas{

template< int dim>
class Compliance
{
public:
  Compliance (){};

  virtual
  dealii::SymmetricTensor<4,dim>
  value(const dealii::Point<dim> &p) const;

  virtual
  void
  value_list (const std::vector<dealii::Point<dim> > & points,
	      std::vector<dealii::SymmetricTensor<4,dim> > & values) const;

private:
  const double poissonsRatio = 0.5;
};


} // End namespace elas

#endif
