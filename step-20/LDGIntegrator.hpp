#ifndef H_LDG_INTEGRATOR__
#define H_LDG_INTEGRATOR__

#include "globals.hpp"

#include <deal.II/base/point.h>
#include <deal.II/base/types.h>
#include <deal.II/base/quadrature.h>
#include <deal.II/meshworker/local_integrator.h>
#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/lapack_full_matrix.h>

#include <map>

namespace elas{
namespace LDGIntegrator{

template <int dim>
class LDGIntegrator: public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  LDGIntegrator
  (dealii::Tensor<1,dim> referenceDirection_In,
   const dealii::Quadrature<dim-1> & face_quadrature_In,
   const dealii::UpdateFlags & update_flags,
   const dealii::MappingQ<dim,dim> & mapping_In,
   std::map<dealii::types::boundary_id, elas::myBoundaryID> const * const BoundaryIDMap);

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


  // void
  // face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
  //      dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
  //      dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
  //      dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const;

  // void
  // boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
  // 	   dealii::MeshWorker::IntegrationInfo<dim> &info) const;

private:
  const dealii::Point<dim> referenceDirection;
  const dealii::SmartPointer<const dealii::Quadrature<dim-1> > face_quadrature;
  const dealii::UpdateFlags face_update_flags;
  const dealii::MappingQ<dim,dim> mapping;
  std::map<dealii::types::boundary_id, elas::myBoundaryID> const * const BoundaryIDMap;
    
};

} // End namespace LDGIntegrator
} // End namespace elas

#endif
