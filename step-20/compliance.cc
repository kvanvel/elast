#include "compliance.hpp"

namespace elas{

template<int dim>
dealii::SymmetricTensor<4,dim> 
Compliance<dim>::value (const dealii::Point<dim> &p )const
{
  const double lambda = 1.4e11;
  const double mu = 79.3e9;
  const double G = mu;
  const double factor1 = (G+lambda)/ (G+G+dim * lambda);
  const double factor2 = factor1 * (G+1.0)/(2.0 * G);

  //const elas::real factor1 = 1.0;
  //const elas::real factor2 = 1.0;
  dealii::SymmetricTensor<4,dim> value;
  
  for (unsigned int i=0; i<dim; ++i)
    for (unsigned int j=0; j<dim; ++j)
      for (unsigned int k=0; k<dim; ++k)
        for (unsigned int l=0; l<dim; ++l)
          value[i][j][k][l] = (((i==k) && (j==l) ? factor2 : 0.0) +
			       ((i==l) && (j==k) ? factor2 : 0.0) +
			       ((i==j) && (k==l) ? factor1 : 0.0));	
  return value;
}

template <int dim>
void
Compliance<dim>::value_list(const std::vector<dealii::Point<dim> > & points,
			    std::vector<dealii::SymmetricTensor<4,dim> >    & values) const
{
  Assert( points.size() == values.size(),
	  dealii::ExcDimensionMismatch (points.size(), values.size()));
  for (unsigned int p = 0; p < points.size(); ++p)
    {
      values[p].clear();
      values[p] = Compliance<dim>::value(points[p]);
    }
}

} //End namespace elas

template class elas::Compliance<2>;
template class elas::Compliance<3>;
