import skbuild.platform_specifics.windows
from skbuild import setup
import sys


# using patched scikit-build for VS2019 courtesy of https://github.com/YannickJadoul/scikit-build
def patched_WindowsPlatform_init(self):
    import textwrap
    from skbuild.platform_specifics.windows import WindowsPlatform, CMakeVisualStudioCommandLineGenerator, CMakeVisualStudioIDEGenerator

    super(WindowsPlatform, self).__init__()

    self._vs_help = textwrap.dedent("""
		Building Windows wheels for requires Microsoft Visual Studio 2017 or 2019:
		  https://visualstudio.microsoft.com/vs/
		""").strip()

    supported_vs_years = [("2019", "v141"), ("2017", "v141")]
    for vs_year, vs_toolset in supported_vs_years:
        self.default_generators.extend([
            CMakeVisualStudioCommandLineGenerator("Ninja", vs_year, vs_toolset),
            CMakeVisualStudioIDEGenerator(vs_year, vs_toolset),
            CMakeVisualStudioCommandLineGenerator("NMake Makefiles", vs_year, vs_toolset),
            CMakeVisualStudioCommandLineGenerator("NMake Makefiles JOM", vs_year, vs_toolset)
        ])


skbuild.platform_specifics.windows.WindowsPlatform.__init__ = patched_WindowsPlatform_init


setup(
    cmake_args=["-DPython3_EXECUTABLE=" + sys.executable],  # scikit-build doesn't detect Python properly on its own
                                                            # in GitHub windows-2019 image using cibuildwheel
    packages=['psnr_hvsm']
)
