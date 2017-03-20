#ifndef __H_LDG__
#define __H_LDG__

#include "globals.hpp"

#include <deal.II/base/config.h>
#include <deal.II/base/exceptions.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/meshworker/dof_info.h>


//#include "SourceBodyValues.hpp"
//#include "DirichletBoundaryValues.hpp"
//#include "NeumannBoundaryValues.hpp"

namespace elas{
namespace LocalIntegrators{
namespace LDG{

template<int dim>
void
massSigma( dealii::FullMatrix<elas::real> & tau_in_sigma_in,
	   const dealii::FEValuesBase<dim> & fe_sigma_Self,
	   const elas::real factor);



template <int dim>
void
StiffSigmaFromU
( dealii::FullMatrix<elas::real> & tau_in_U_in,
  const dealii::FEValuesBase<dim> & fe_tau_Self,
  const dealii::FEValuesBase<dim> & fe_U_Self,
  const elas::real factor);

template <int dim>
void
numericalFluxSigmaFromU
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 dealii::FullMatrix<double> & tau_in_U_out,
 dealii::FullMatrix<double> & tau_out_U_in,
 dealii::FullMatrix<double> & tau_out_U_out,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_tau_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig
 );

template <int dim>
void
numericalFluxUFromSigma
(
 const dealii::Point<dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 dealii::FullMatrix<double> & V_in_sigma_out,
 dealii::FullMatrix<double> & V_out_sigma_in,
 dealii::FullMatrix<double> & V_out_sigma_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Neig
 );


} //End namespace LDG
} //End namespace LocalIntegrators
} //End namespace elas  

//Implementation file.
#include "LDG.tcc"

#endif
