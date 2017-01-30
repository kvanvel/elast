#include "NueBC.hpp"

namespace elas{

template<int dim>
dealii::SymmetricTensor<2,dim>
NueBC<dim>::value(const dealii::Point<dim> &p) const
{
  dealii::SymmetricTensor<2,dim> value;
  for(unsigned int i = 0; i < dim; ++i){
    value[i][i] = 0.0;
  }
  return value;
}

template<int dim>
void
NueBC<dim>::value_list(const std::vector<dealii::Point<dim> > & points,
		       std::vector<dealii::SymmetricTensor<2,dim> > & values) const
{
  Assert( points.size() == values.size(),
	  dealii::ExcDimensionMismatch (points.size(), values.size()));
  for (unsigned int p = 0; p < points.size(); ++p){
    values[p].clear();
    values[p] = elas::NueBC<dim>::value(points[p]);
  }
}



}// End Namespace elas


template class elas::NueBC<2>;
template class elas::NueBC<3>;
