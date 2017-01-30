#ifndef H_LDG_ERROR_INTEGRATOR__
#define H_LDG_ERROR_INTEGRATOR__

#include "globals.hpp"

#include <deal.II/fe/mapping_q.h>
#include <deal.II/integrators/divergence.h>
#include <deal.II/integrators/l2.h>
#include <deal.II/integrators/laplace.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/loop.h>
#include <deal.II/meshworker/output.h>
#include <deal.II/meshworker/simple.h>

namespace elas{
namespace LDGErrorIntegrator{

template <int dim>
class LDGErrorIntegrator: public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  LDGErrorIntegrator
  (dealii::Point<dim> referenceDirection_In);
   

  void
  cell(dealii::MeshWorker::DoFInfo<dim> & dinfo,
       dealii::MeshWorker::IntegrationInfo<dim> &info) const;

  void
  face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
       dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
       dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
       dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const;

  void
  boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
	   dealii::MeshWorker::IntegrationInfo<dim> & info) const;

private:
    
};

} // End namespace LDGIntegrator
} // End namespace elas

#endif
