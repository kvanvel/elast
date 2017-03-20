#include "elasRHS.hpp"

namespace elas{

template <int dim>
dealii::Tensor<1,dim>
elasRHS<dim>::value (const dealii::Point<dim> &p) const
{
  dealii::Tensor<1,dim>  value;
  value[0] = 0.0;
  for(unsigned int i = 1; i<dim; ++i){
    value[i] = 1.0+0.8*std::sin(8*dealii::numbers::PI*p[0]);
    value[i] = 1.0;
    value[i] = 0;
  }
  return value;
}

template <int dim>
void
elasRHS<dim>::value_list(const std::vector<dealii::Point<dim> > &points,
			 std::vector<dealii::Tensor<1,dim> > &values) const
{
  Assert(values.size() == points.size(),
	 dealii::ExcDimensionMismatch (values.size(), points.size() ) );

  for(unsigned int i = 0; i < points.size(); ++i){
    values[i] = elasRHS<dim>::value(points[i]);
  }
}


}//end namespace elas

template class elas::elasRHS<2>;
template class elas::elasRHS<3>;

