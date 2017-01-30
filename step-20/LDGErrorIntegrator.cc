#include "LDGErrorIntegrator.hpp"
#include "LDG.hpp"

namespace elas{
namespace LDGErrorIntegrator{

template<int dim>
LDGErrorIntegrator<dim>::LDGErrorIntegrator
(dealii::Point<dim> referenceDirection_In)
  :
  dealii::MeshWorker::LocalIntegrator<dim>(true, true, true)  
{}

template<int dim>
void
LDGErrorIntegrator<dim>::cell
(dealii::MeshWorker::DoFInfo<dim> & dinfo_Self,
 dealii::MeshWorker::IntegrationInfo<dim> &infoSelf ) const{

  elas::real normSquaredOfCellResidual = 0;

  std::vector<dealii::types::global_dof_index>
    local_dof_indices_Sigma(infoSelf.fe_values(0).dofs_per_cell),
    local_dof_indices_U    (infoSelf.fe_values(1).dofs_per_cell);

  dealii::Vector<elas::real>
    local_dofs_Sigma(local_dof_indices_Sigma.size() ),
    local_dofs_U    (local_dof_indices_U.size() );
  
  const auto Indices_By_Block = dinfo_Self.indices_by_block;
    
  const auto & Sigma_Indices = Indices_By_Block[0];
  const auto & U_Indices     = Indices_By_Block[1];    
    
  std::shared_ptr<dealii::MeshWorker::VectorDataBase<dim,dim> >
    GlobalDataPtr = infoSelf.global_data;

  const auto & globalDataCollection = GlobalDataPtr->data;

  const auto & SolutionVectorPtr
    = globalDataCollection.template read_ptr<dealii::BlockVector<elas::real> > ("solution");
    
  
  SolutionVectorPtr->extract_subvector_to(local_dof_indices_Sigma.begin(),
					  local_dof_indices_Sigma.end(),
					  local_dofs_Sigma.begin() );

  for(unsigned int i = 0; i < local_dofs_Sigma.size(); ++i){    
    //std::cout << "local_dofs_Sigma[" << i << "] = " << local_dofs_Sigma[i] << std::endl;
  }

  SolutionVectorPtr->extract_subvector_to(local_dof_indices_U.begin(),
					  local_dof_indices_U.end(),
					  local_dofs_U.begin() );
      
    
  elas::
    LocalIntegrators::
    LDG::
    CellResidualSigma(infoSelf.fe_values(0),
		      local_dofs_Sigma,
		      normSquaredOfCellResidual);
  

  dinfo_Self.value(0) =
    normSquaredOfCellResidual
    * dinfo_Self.cell->diameter()
    * dinfo_Self.cell->diameter();
}


template <int dim>
void
LDGErrorIntegrator<dim>::face(dealii::MeshWorker::DoFInfo<dim> & dinfoSELF,
			      dealii::MeshWorker::DoFInfo<dim> & dinfoNEIG,
			      dealii::MeshWorker::IntegrationInfo<dim> &infoSELF,
			      dealii::MeshWorker::IntegrationInfo<dim> &infoNEIG ) const {
  elas::real jumpSquared = 0;


  // std::vector<dealii::types::global_dof_index>
  //   local_dof_indices_Sigma_Self(infoSELF.fe_values(0).dofs_per_cell),
  //   local_dof_indices_Sigma_Neig(infoNEIG.fe_values(0).dofs_per_cell),
  //   local_dof_indices_U_Self(infoSELF.fe_values(1).dofs_per_cell),
  //   local_dof_indices_U_Neig(infoNEIG.fe_values(1).dofs_per_cell);


  // std::vector<elas::real>
  //   std_Local_dofs_Sigma_Self(local_dof_indices_Sigma_Self.size() );

  //local_dofs_Sigma_Self = 0;
  


  const auto Indices_By_Block_Self = dinfoSELF.indices_by_block;
  const auto Indices_By_Block_Neig = dinfoNEIG.indices_by_block;

  const auto &
    Sigma_Indices_Self = Indices_By_Block_Self[0],
    Sigma_Indices_Neig = Indices_By_Block_Neig[0],
    U_Indices_Self = Indices_By_Block_Self[1],
    U_Indices_Neig = Indices_By_Block_Neig[1];

    dealii::Vector<elas::real>
      local_dofs_Sigma_Self(Sigma_Indices_Self.size()),
      local_dofs_Sigma_Neig(Sigma_Indices_Neig.size()),      
      local_dofs_U_Self(U_Indices_Self.size()),      
      local_dofs_U_Neig(U_Indices_Neig.size());


  std::shared_ptr<dealii::MeshWorker::VectorDataBase<dim,dim> >
    GlobalDataPtr_Self = infoSELF.global_data,
    GlobalDataPtr_Neig = infoNEIG.global_data;

  const auto & globalDataCollection = GlobalDataPtr_Self->data;

  const auto SolutionVectorPtr
    = globalDataCollection.template read_ptr<dealii::BlockVector<elas::real> > ("solution");


  SolutionVectorPtr->extract_subvector_to(Sigma_Indices_Self.begin(),
					  Sigma_Indices_Self.end(),
					  local_dofs_Sigma_Self.begin());
  

  SolutionVectorPtr->extract_subvector_to(Sigma_Indices_Neig.begin(),
					  Sigma_Indices_Neig.end(),
					  local_dofs_Sigma_Neig.begin());


  SolutionVectorPtr->extract_subvector_to(U_Indices_Self.begin(),
					  U_Indices_Self.end(),		  
					  local_dofs_U_Self.begin());


  SolutionVectorPtr->extract_subvector_to(U_Indices_Neig.begin(),
					  U_Indices_Neig.end(),		  
					  local_dofs_U_Neig.begin());

  
  elas::LocalIntegrators::LDG::SigmaJump(infoSELF.fe_values(0),
					 infoNEIG.fe_values(0),
					 local_dofs_Sigma_Self,
					 local_dofs_Sigma_Neig,
					 jumpSquared);

  dinfoSELF.value(0) = jumpSquared * dinfoSELF.cell->diameter();
  dinfoNEIG.value(0) = jumpSquared * dinfoSELF.cell->diameter();
    
  //PRINT(jumpSquared);
  
  
}
  
template<int dim>
void
LDGErrorIntegrator<dim>::boundary(dealii::MeshWorker::DoFInfo<dim> & dinfo,
				  dealii::MeshWorker::IntegrationInfo<dim> & info) const
{}

} // End Namespace LDGErrorIntegrator
} // End Namespace heat

template class elas::LDGErrorIntegrator::LDGErrorIntegrator<2>;
template class elas::LDGErrorIntegrator::LDGErrorIntegrator<3>;
