#ifndef LDG_INTEGRATOR_
#define LDG_INTEGRATOR_

#include "globals.hpp"



namespace stokes{
namespace LDGIntegrator {

template<int dim>
class LDGIntegrator : public dealii::MeshWorker::LocalIntegrator<dim>
{
public:
  LDGIntegrator(const dealii::Tensor<1,dim> & referenceDirection,
		const dealii::Quadrature<dim-1> & face_quadrature,
		const dealii::UpdateFlags & update_flags,
		const dealii::MappingQ<dim,dim> & mapping_In,
		const std::map<dealii::types::boundary_id, stokes::myBoundaryID> & BoundaryIDMap);

  void
  cell(dealii::MeshWorker::DoFInfo<dim> & dinfo,
       dealii::MeshWorker::IntegrationInfo<dim> & info) const;

  void
  face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
       dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
       dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
       dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const;

  void
  boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
	   dealii::MeshWorker::IntegrationInfo<dim> & info) const;

private:
  const dealii::Tensor<1,dim> referenceDirection;
  const dealii::SmartPointer<const dealii::Quadrature<dim-1> > face_quadrature;
  const dealii::UpdateFlags face_update_flags;
  const dealii::MappingQ<dim,dim> mapping;
  const std::map<dealii::types::boundary_id, stokes::myBoundaryID> BoundaryIDMap;
  
};

} //end namespace
} //end namespace

#include "LDGIntegrator.tpp"

#endif
