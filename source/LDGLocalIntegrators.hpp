#ifndef LDG_LOCAL_INTEGRATORS_
#define LDG_LOCAL_INTEGRATORS_

#include "globals.hpp"

namespace stokes {
namespace LocalIntegrators {
namespace LDG {

template<int dim>
void
massSigma( dealii::FullMatrix<stokes::real> & tau_in_sigma_in,
	   const dealii::FEValuesBase<dim> & fe_sigma_Self,
	   const stokes::real factor);

template <int dim>
void
StiffSigmaFromU
( dealii::FullMatrix<stokes::real> & tau_in_U_in,
  const dealii::FEValuesBase<dim> & fe_tau_Self,
  const dealii::FEValuesBase<dim> & fe_U_Self,
  const stokes::real factor);

template <int dim>
void
StiffUFromSigma
( dealii::FullMatrix<stokes::real> & V_in_sigma_in,
  const dealii::FEValuesBase<dim> & fe_V_Self,
  const dealii::FEValuesBase<dim> & fe_sigma_Self,
  const stokes::real factor);


template <int dim>
void
StiffUFromP(dealii::FullMatrix<stokes::real> & V_in_P_in,
	    const dealii::FEValuesBase<dim> & fe_V_Self,
	    const dealii::FEValuesBase<dim> & fe_P_Self,
	    const stokes::real factor);


template <int dim>
void
StiffUFromP(dealii::FullMatrix<stokes::real> & V_in_P_in,
	    const dealii::FEValuesBase<dim> & fe_V_Self,
	    const dealii::FEValuesBase<dim> & fe_P_Self,
	    const stokes::real factor);


template <int dim>
void
StiffPFromU(dealii::FullMatrix<stokes::real> & P_in_U_in,
	    const dealii::FEValuesBase<dim> & fe_R_Self,
	    const dealii::FEValuesBase<dim> & fe_U_Self,
	    const stokes::real factor);

template <int dim>
void
numericalTraceSigmaFromU
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 dealii::FullMatrix<double> & tau_in_U_out,
 dealii::FullMatrix<double> & tau_out_U_in,
 dealii::FullMatrix<double> & tau_out_U_out,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_tau_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig,
 const stokes::real factor);

  
template <int dim>
void
numericalTraceSigmaFromUBoundary
(const dealii::Tensor<1,dim> & referenceDirection,
 dealii::FullMatrix<double> & tau_in_U_in,
 const dealii::FEValuesBase<dim> & fe_tau_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const stokes::real factor);

template <int dim>
void
numericalTraceUFromSigma
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 dealii::FullMatrix<double> & V_in_sigma_out,
 dealii::FullMatrix<double> & V_out_sigma_in,
 dealii::FullMatrix<double> & V_out_sigma_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Neig,
 const stokes::real factor);


template <int dim>
void
numericalTraceUFromSigmaBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_sigma_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_sigma_Self,
 const stokes::real factor);



template <int dim>
void
numericalTraceUFromP
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_P_in,
 dealii::FullMatrix<double> & V_in_P_out,
 dealii::FullMatrix<double> & V_out_P_in,
 dealii::FullMatrix<double> & V_out_P_out,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_V_Neig,
 const dealii::FEValuesBase<dim> & fe_P_Self,
 const dealii::FEValuesBase<dim> & fe_P_Neig,
 const stokes::real factor);

template <int dim>
void
numericalTraceUFromPBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & V_in_P_in,
 const dealii::FEValuesBase<dim> & fe_V_Self,
 const dealii::FEValuesBase<dim> & fe_P_Self,
 const stokes::real factor);


template <int dim>
void
numericalTracePFromU
(
 const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & R_in_U_in,
 dealii::FullMatrix<double> & R_in_U_out,
 dealii::FullMatrix<double> & R_out_U_in,
 dealii::FullMatrix<double> & R_out_U_out,
 const dealii::FEValuesBase<dim> & fe_R_Self,
 const dealii::FEValuesBase<dim> & fe_R_Neig,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const dealii::FEValuesBase<dim> & fe_U_Neig,
 const stokes::real factor);

template <int dim>
void
numericalTracePFromUBoundary
(const dealii::Tensor<1,dim> referenceDirection,
 dealii::FullMatrix<double> & R_in_U_in,
 const dealii::FEValuesBase<dim> & fe_R_Self,
 const dealii::FEValuesBase<dim> & fe_U_Self,
 const stokes::real factor);


  






} //end namespace
} //end namespace
} //end namespace

#include "LDGLocalIntegrators.tpp"

#endif
