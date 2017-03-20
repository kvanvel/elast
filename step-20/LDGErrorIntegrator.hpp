#ifndef H_LDG_ERROR_INTEGRATOR__
#define H_LDG_ERROR_INTEGRATOR__

#include <deal.II/base/point.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/local_integrator.h>

namespace elas{
namespace LDGErrorIntegrator{

template <int dim>
class LDGErrorIntegrator: public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  LDGErrorIntegrator ();   

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
