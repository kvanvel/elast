#include "DirBC.hpp"

namespace elas{

template <int dim>
dealii::Tensor<1,dim>
DirBC<dim>::value (const dealii::Point<dim> &p) const
{
  dealii::Tensor<1,dim> value;
  for(unsigned int i = 0; i < dim; ++i){
    value[i] = p[i];
  }
  return value;
}

template <int dim>
void
DirBC<dim>::value_list(const std::vector<dealii::Point<dim> > &points,
		       std::vector<dealii::Tensor<1,dim> > &values) const
{
  Assert(values.size() == points.size(),
	 dealii::ExcDimensionMismatch (values.size(), points.size() ) );

  for(unsigned int i = 0; i < points.size(); ++i){
    values[i] = DirBC<dim>::value(points[i]);
  }
}

} // end namespace elas


template class elas::DirBC<2>;
template class elas::DirBC<3>;
