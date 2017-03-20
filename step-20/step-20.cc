#include "globals.hpp"

#include "DirBC.hpp"
#include "NueBC.hpp"
#include "elasRHS.hpp"
#include "compliance.hpp"
#include "LDGIntegrator.hpp"
#include "LDGErrorIntegrator.hpp"
#include "elasMixedLaplaceProblem.hpp"

int main ()
{
  try
    {
      using namespace dealii;
      //using namespace Step20;

      deallog.depth_console (-1);

      elas::elasMixedLaplaceProblem<2> mixed_laplace_problem(4);
      mixed_laplace_problem.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
