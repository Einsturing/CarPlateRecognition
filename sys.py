# encoding: utf-8
# module sys
# from (built-in)
# by generator 1.147
"""
This module provides access to some objects used or maintained by the
interpreter and to functions that interact strongly with the interpreter.

Dynamic objects:

argv -- command line arguments; argv[0] is the script pathname if known
path -- module search path; path[0] is the script directory, else ''
modules -- dictionary of loaded modules

displayhook -- called to show results in an interactive session
excepthook -- called to handle any uncaught exception other than SystemExit
  To customize printing in an interactive session or to install a custom
  top-level exception handler, assign other functions to replace these.

stdin -- standard input file object; used by input()
stdout -- standard output file object; used by print()
stderr -- standard error object; used for error messages
  By assigning other file objects (or objects that behave like files)
  to these, it is possible to redirect all of the interpreter's I/O.

last_type -- type of last uncaught exception
last_value -- value of last uncaught exception
last_traceback -- traceback of last uncaught exception
  These three are only available in an interactive session after a
  traceback has been printed.

Static objects:

builtin_module_names -- tuple of module names built into this interpreter
copyright -- copyright notice pertaining to this interpreter
exec_prefix -- prefix used to find the machine-specific Python library
executable -- absolute path of the executable binary of the Python interpreter
float_info -- a named tuple with information about the float implementation.
float_repr_style -- string indicating the style of repr() output for floats
hash_info -- a named tuple with information about the hash algorithm.
hexversion -- version information encoded as a single integer
implementation -- Python implementation information.
int_info -- a named tuple with information about the int implementation.
maxsize -- the largest supported length of containers.
maxunicode -- the value of the largest Unicode code point
platform -- platform identifier
prefix -- prefix used to find the Python library
thread_info -- a named tuple with information about the thread implementation.
version -- the version of this interpreter as a string
version_info -- version information as a named tuple
dllhandle -- [Windows only] integer handle of the Python DLL
winver -- [Windows only] version number of the Python DLL
_enablelegacywindowsfsencoding -- [Windows only]
__stdin__ -- the original stdin; don't touch!
__stdout__ -- the original stdout; don't touch!
__stderr__ -- the original stderr; don't touch!
__displayhook__ -- the original displayhook; don't touch!
__excepthook__ -- the original excepthook; don't touch!

Functions:

displayhook() -- print an object to the screen, and save it in builtins._
excepthook() -- print an exception and its traceback to sys.stderr
exc_info() -- return thread-safe information about the current exception
exit() -- exit the interpreter by raising SystemExit
getdlopenflags() -- returns flags to be used for dlopen() calls
getprofile() -- get the global profiling function
getrefcount() -- return the reference count for an object (plus one :-)
getrecursionlimit() -- return the max recursion depth for the interpreter
getsizeof() -- return the size of an object in bytes
gettrace() -- get the global debug tracing function
setcheckinterval() -- control how often the interpreter checks for events
setdlopenflags() -- set the flags to be used for dlopen() calls
setprofile() -- set the global profiling function
setrecursionlimit() -- set the max recursion depth for the interpreter
settrace() -- set the global debug tracing function
"""
# no imports

# Variables with simple values

api_version = 1013

base_exec_prefix = 'C:\\ProgramData\\Anaconda3'

base_prefix = 'C:\\ProgramData\\Anaconda3'

byteorder = 'little'

copyright = 'Copyright (c) 2001-2020 Python Software Foundation.\nAll Rights Reserved.\n\nCopyright (c) 2000 BeOpen.com.\nAll Rights Reserved.\n\nCopyright (c) 1995-2001 Corporation for National Research Initiatives.\nAll Rights Reserved.\n\nCopyright (c) 1991-1995 Stichting Mathematisch Centrum, Amsterdam.\nAll Rights Reserved.'

dllhandle = 140712791048192

dont_write_bytecode = True

executable = 'C:\\ProgramData\\Anaconda3\\python.exe'

exec_prefix = 'C:\\ProgramData\\Anaconda3'

float_repr_style = 'short'

hexversion = 50856944

maxsize = 9223372036854775807
maxunicode = 1114111

platform = 'win32'

prefix = 'C:\\ProgramData\\Anaconda3'

pycache_prefix = None

version = '3.8.3 (default, Jul  2 2020, 17:30:36) [MSC v.1916 64 bit (AMD64)]'

winver = '3.8'

_base_executable = 'C:\\ProgramData\\Anaconda3\\python.exe'

_framework = ''

_home = None


# functions

def addaudithook(*args, **kwargs):  # real signature unknown
    """ Adds a new audit hook callback. """
    pass


def audit(event, *args):  # real signature unknown; restored from __doc__
    """
    audit(event, *args)

    Passes the event to any audit hooks that are attached.
    """
    pass


def breakpointhook(*args, **kws):  # real signature unknown; restored from __doc__
    """
    breakpointhook(*args, **kws)

    This hook function is called by built-in breakpoint().
    """
    pass


def callstats(*args, **kwargs):  # real signature unknown
    """
    Return a tuple of function call statistics.

    A tuple is returned only if CALL_PROFILE was defined when Python was
    built.  Otherwise, this returns None.

    When enabled, this function returns detailed, implementation-specific
    details about the number of function calls executed. The return value
    is a 11-tuple where the entries in the tuple are counts of:
    0. all function calls
    1. calls to PyFunction_Type objects
    2. PyFunction calls that do not create an argument tuple
    3. PyFunction calls that do not create an argument tuple
       and bypass PyEval_EvalCodeEx()
    4. PyMethod calls
    5. PyMethod calls on bound methods
    6. PyType calls
    7. PyCFunction calls
    8. generator calls
    9. All other calls
    10. Number of stack pops performed by call_function()
    """
    pass


def call_tracing(*args, **kwargs):  # real signature unknown
    """
    Call func(*args), while tracing is enabled.

    The tracing state is saved, and restored afterwards.  This is intended
    to be called from a debugger from a checkpoint, to recursively debug
    some other code.
    """
    pass


def displayhook(*args, **kwargs):  # real signature unknown
    """ Print an object to sys.stdout and also save it in builtins._ """
    pass


def excepthook(*args, **kwargs):  # real signature unknown
    """ Handle an exception by displaying it with a traceback on sys.stderr. """
    pass


def exc_info(*args, **kwargs):  # real signature unknown
    """
    Return current exception information: (type, value, traceback).

    Return information about the most recent exception caught by an except
    clause in the current stack frame or in an older stack frame.
    """
    pass


def exit(*args, **kwargs):  # real signature unknown
    """
    Exit the interpreter by raising SystemExit(status).

    If the status is omitted or None, it defaults to zero (i.e., success).
    If the status is an integer, it will be used as the system exit status.
    If it is another kind of object, it will be printed and the system
    exit status will be one (i.e., failure).
    """
    pass


def getallocatedblocks(*args, **kwargs):  # real signature unknown
    """ Return the number of memory blocks currently allocated. """
    pass


def getcheckinterval(*args, **kwargs):  # real signature unknown
    """ Return the current check interval; see sys.setcheckinterval(). """
    pass


def getdefaultencoding(*args, **kwargs):  # real signature unknown
    """ Return the current default encoding used by the Unicode implementation. """
    pass


def getfilesystemencodeerrors(*args, **kwargs):  # real signature unknown
    """ Return the error mode used Unicode to OS filename conversion. """
    pass


def getfilesystemencoding(*args, **kwargs):  # real signature unknown
    """ Return the encoding used to convert Unicode filenames to OS filenames. """
    pass


def getprofile(*args, **kwargs):  # real signature unknown
    """
    Return the profiling function set with sys.setprofile.

    See the profiler chapter in the library manual.
    """
    pass


def getrecursionlimit(*args, **kwargs):  # real signature unknown
    """
    Return the current value of the recursion limit.

    The recursion limit is the maximum depth of the Python interpreter
    stack.  This limit prevents infinite recursion from causing an overflow
    of the C stack and crashing Python.
    """
    pass


def getrefcount():  # real signature unknown; restored from __doc__
    """
    Return the reference count of object.

    The count returned is generally one higher than you might expect,
    because it includes the (temporary) reference as an argument to
    getrefcount().
    """
    pass


def getsizeof(p_object, default=None):  # real signature unknown; restored from __doc__
    """
    getsizeof(object [, default]) -> int

    Return the size of object in bytes.
    """
    return 0


def getswitchinterval(*args, **kwargs):  # real signature unknown
    """ Return the current thread switch interval; see sys.setswitchinterval(). """
    pass


def gettrace(*args, **kwargs):  # real signature unknown
    """
    Return the global debug tracing function set with sys.settrace.

    See the debugger chapter in the library manual.
    """
    pass


def getwindowsversion(*args, **kwargs):  # real signature unknown
    """
    Return info about the running version of Windows as a named tuple.

    The members are named: major, minor, build, platform, service_pack,
    service_pack_major, service_pack_minor, suite_mask, product_type and
    platform_version. For backward compatibility, only the first 5 items
    are available by indexing. All elements are numbers, except
    service_pack and platform_type which are strings, and platform_version
    which is a 3-tuple. Platform is always 2. Product_type may be 1 for a
    workstation, 2 for a domain controller, 3 for a server.
    Platform_version is a 3-tuple containing a version number that is
    intended for identifying the OS rather than feature detection.
    """
    pass


def get_asyncgen_hooks(*args, **kwargs):  # real signature unknown
    """
    Return the installed asynchronous generators hooks.

    This returns a namedtuple of the form (firstiter, finalizer).
    """
    pass


def get_coroutine_origin_tracking_depth(*args, **kwargs):  # real signature unknown
    """ Check status of origin tracking for coroutine objects in this thread. """
    pass


def intern(*args, **kwargs):  # real signature unknown
    """
    ``Intern'' the given string.

    This enters the string in the (global) table of interned strings whose
    purpose is to speed up dictionary lookups. Return the string itself or
    the previously interned string object with the same value.
    """
    pass


def is_finalizing(*args, **kwargs):  # real signature unknown
    """ Return True if Python is exiting. """
    pass


def setcheckinterval(*args, **kwargs):  # real signature unknown
    """
    Set the async event check interval to n instructions.

    This tells the Python interpreter to check for asynchronous events
    every n instructions.

    This also affects how often thread switches occur.
    """
    pass


def setprofile(function):  # real signature unknown; restored from __doc__
    """
    setprofile(function)

    Set the profiling function.  It will be called on each function call
    and return.  See the profiler chapter in the library manual.
    """
    pass


def setrecursionlimit(*args, **kwargs):  # real signature unknown
    """
    Set the maximum depth of the Python interpreter stack to n.

    This limit prevents infinite recursion from causing an overflow of the C
    stack and crashing Python.  The highest possible limit is platform-
    dependent.
    """
    pass


def setswitchinterval(*args, **kwargs):  # real signature unknown
    """
    Set the ideal thread switching delay inside the Python interpreter.

    The actual frequency of switching threads can be lower if the
    interpreter executes long sequences of uninterruptible code
    (this is implementation-specific and workload-dependent).

    The parameter must represent the desired switching delay in seconds
    A typical value is 0.005 (5 milliseconds).
    """
    pass


def settrace(function):  # real signature unknown; restored from __doc__
    """
    settrace(function)

    Set the global debug tracing function.  It will be called on each
    function call.  See the debugger chapter in the library manual.
    """
    pass


def set_asyncgen_hooks(*args, **kwargs):  # real signature unknown; NOTE: unreliably restored from __doc__
    """
    set_asyncgen_hooks(* [, firstiter] [, finalizer])

    Set a finalizer for async generators objects.
    """
    pass


def set_coroutine_origin_tracking_depth(*args, **kwargs):  # real signature unknown
    """
    Enable or disable origin tracking for coroutine objects in this thread.

    Coroutine objects will track 'depth' frames of traceback information
    about where they came from, available in their cr_origin attribute.

    Set a depth of 0 to disable.
    """
    pass


def unraisablehook(*args, **kwargs):  # real signature unknown
    """
    Handle an unraisable exception.

    The unraisable argument has the following attributes:

    * exc_type: Exception type.
    * exc_value: Exception value, can be None.
    * exc_traceback: Exception traceback, can be None.
    * err_msg: Error message, can be None.
    * object: Object causing the exception, can be None.
    """
    pass


def _clear_type_cache(*args, **kwargs):  # real signature unknown
    """ Clear the internal type lookup cache. """
    pass


def _current_frames(*args, **kwargs):  # real signature unknown
    """
    Return a dict mapping each thread's thread id to its current stack frame.

    This function should be used for specialized purposes only.
    """
    pass


def _debugmallocstats(*args, **kwargs):  # real signature unknown
    """
    Print summary info to stderr about the state of pymalloc's structures.

    In Py_DEBUG mode, also perform some expensive internal consistency
    checks.
    """
    pass


def _enablelegacywindowsfsencoding(*args, **kwargs):  # real signature unknown
    """
    Changes the default filesystem encoding to mbcs:replace.

    This is done for consistency with earlier versions of Python. See PEP
    529 for more information.

    This is equivalent to defining the PYTHONLEGACYWINDOWSFSENCODING
    environment variable before launching Python.
    """
    pass


def _getframe(*args, **kwargs):  # real signature unknown
    """
    Return a frame object from the call stack.

    If optional integer depth is given, return the frame object that many
    calls below the top of the stack.  If that is deeper than the call
    stack, ValueError is raised.  The default for depth is zero, returning
    the frame at the top of the call stack.

    This function should be used for internal and specialized purposes
    only.
    """
    pass


def __breakpointhook__(*args, **kwargs):  # real signature unknown
    """
    breakpointhook(*args, **kws)

    This hook function is called by built-in breakpoint().
    """
    pass


def __displayhook__(*args, **kwargs):  # real signature unknown
    """ Print an object to sys.stdout and also save it in builtins._ """
    pass


def __excepthook__(*args, **kwargs):  # real signature unknown
    """ Handle an exception by displaying it with a traceback on sys.stderr. """
    pass


def __interactivehook__():  # reliably restored by inspect
    # no doc
    pass


def __unraisablehook__(*args, **kwargs):  # real signature unknown
    """
    Handle an unraisable exception.

    The unraisable argument has the following attributes:

    * exc_type: Exception type.
    * exc_value: Exception value, can be None.
    * exc_traceback: Exception traceback, can be None.
    * err_msg: Error message, can be None.
    * object: Object causing the exception, can be None.
    """
    pass


# classes

class __loader__(object):
    """
    Meta path import for built-in modules.

        All methods are either class or static methods to avoid the need to
        instantiate the class.
    """

    @classmethod
    def create_module(cls, *args, **kwargs):  # real signature unknown
        """ Create a built-in module """
        pass

    @classmethod
    def exec_module(cls, *args, **kwargs):  # real signature unknown
        """ Exec a built-in module """
        pass

    @classmethod
    def find_module(cls, *args, **kwargs):  # real signature unknown
        """
        Find the built-in module.

                If 'path' is ever specified then the search is considered a failure.

                This method is deprecated.  Use find_spec() instead.
        """
        pass

    @classmethod
    def find_spec(cls, *args, **kwargs):  # real signature unknown
        pass

    @classmethod
    def get_code(cls, *args, **kwargs):  # real signature unknown
        """ Return None as built-in modules do not have code objects. """
        pass

    @classmethod
    def get_source(cls, *args, **kwargs):  # real signature unknown
        """ Return None as built-in modules do not have source code. """
        pass

    @classmethod
    def is_package(cls, *args, **kwargs):  # real signature unknown
        """ Return False as built-in modules are never packages. """
        pass

    @classmethod
    def load_module(cls, *args, **kwargs):  # real signature unknown
        """
        Load the specified module into sys.modules and return it.

            This method is deprecated.  Use loader.exec_module instead.
        """
        pass

    def module_repr(module):  # reliably restored by inspect
        """
        Return repr for the module.

                The method is deprecated.  The import machinery does the job itself.
        """
        pass

    def __init__(self, *args, **kwargs):  # real signature unknown
        pass

    __weakref__ = property(lambda self: object(), lambda self, v: None, lambda self: None)  # default
    """list of weak references to the object (if defined)"""

    __dict__ = None  # (!) real value is "mappingproxy({'__module__': '_frozen_importlib', '__doc__': 'Meta path import for built-in modules.\\n\\n    All methods are either class or static methods to avoid the need to\\n    instantiate the class.\\n\\n    ', 'module_repr': <staticmethod object at 0x000001706EDF3460>, 'find_spec': <classmethod object at 0x000001706EDF3490>, 'find_module': <classmethod object at 0x000001706EDF34C0>, 'create_module': <classmethod object at 0x000001706EDF34F0>, 'exec_module': <classmethod object at 0x000001706EDF3520>, 'get_code': <classmethod object at 0x000001706EDF35B0>, 'get_source': <classmethod object at 0x000001706EDF3640>, 'is_package': <classmethod object at 0x000001706EDF36D0>, 'load_module': <classmethod object at 0x000001706EDF3700>, '__dict__': <attribute '__dict__' of 'BuiltinImporter' objects>, '__weakref__': <attribute '__weakref__' of 'BuiltinImporter' objects>})"


# variables with complex values

argv = []  # real value of type <class 'list'> skipped

builtin_module_names = ()  # real value of type <class 'tuple'> skipped

flags = (
    0,
    0,
    0,
    0,
    1,
    0,
    0,
    0,
    0,
    0,
    0,
    1,
    0,
    False,
    0,
)

float_info = (
    1.7976931348623157e+308,
    1024,
    308,
    2.2250738585072014e-308,
    -1021,
    -307,
    15,
    53,
    2.220446049250313e-16,
    2,
    1,
)

hash_info = (
    64,
    2305843009213693951,
    314159,
    0,
    1000003,
    'siphash24',
    64,
    128,
    0,
)

implementation = None  # (!) real value is "namespace(cache_tag='cpython-38', hexversion=50856944, name='cpython', version=sys.version_info(major=3, minor=8, micro=3, releaselevel='final', serial=0))"

int_info = (
    30,
    4,
)

meta_path = [
    __loader__,
    None,  # (!) real value is "<class '_frozen_importlib.FrozenImporter'>"
    None,  # (!) real value is "<class '_frozen_importlib_external.PathFinder'>"
    None,  # (!) real value is '<six._SixMetaPathImporter object at 0x000001706FB2BD60>'
]

modules = {}  # real value of type <class 'dict'> skipped

path = [
    'C:\\Program Files\\JetBrains\\PyCharm 2020.3.2\\plugins\\python\\helpers',
    'C:\\Program Files\\JetBrains\\PyCharm 2020.3.2\\plugins\\python\\helpers\\generator3',
    'C:\\ProgramData\\Anaconda3\\python38.zip',
    'C:\\ProgramData\\Anaconda3\\DLLs',
    'C:\\ProgramData\\Anaconda3\\lib',
    'C:\\ProgramData\\Anaconda3',
    'C:\\Users\\11064\\AppData\\Roaming\\Python\\Python38\\site-packages',
    'C:\\ProgramData\\Anaconda3\\lib\\site-packages',
    'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\win32',
    'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\win32\\lib',
    'C:\\ProgramData\\Anaconda3\\lib\\site-packages\\Pythonwin',
    'C:/ProgramData/Anaconda3/DLLs',
    'C:/ProgramData/Anaconda3/Lib',
    'C:/Users/11064/AppData/Roaming/Python/Python38/site-packages',
    'C:/ProgramData/Anaconda3/Lib/site-packages',
    'C:/ProgramData/Anaconda3/Lib/site-packages/win32',
    'C:/ProgramData/Anaconda3/Lib/site-packages/win32/lib',
    'C:/ProgramData/Anaconda3/Lib/site-packages/pythonwin',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stdlib',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/jwt',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/six',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/mock',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/nmap',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/attrs',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/polib',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/retry',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/enum34',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/orjson',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/xxhash',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/chardet',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/futures',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/tzlocal',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/aiofiles',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/filelock',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/Markdown',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/paramiko',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/pathlib2',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/requests',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/waitress',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/Werkzeug',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/freezegun',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/ipaddress',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/pyRFC3339',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/typed-ast',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/cachetools',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/Deprecated',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/frozendict',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/contextvars',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/JACK-Client',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/atomicwrites',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/cryptography',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/click-spinner',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/DateTimeRange',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/pkg_resources',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/python-gflags',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/python-slugify',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/python-dateutil',
    'C:/Program Files/JetBrains/PyCharm 2020.3.2/plugins/python/helpers/typeshed/stubs/typing-extensions',
]

path_hooks = [
    None,  # (!) real value is "<class 'zipimport.zipimporter'>"
    None,  # (!) real value is '<function FileFinder.path_hook.<locals>.path_hook_for_FileFinder at 0x000001706EE26160>'
]

path_importer_cache = {}  # real value of type <class 'dict'> skipped

stderr = None  # (!) real value is "<_io.TextIOWrapper name='<stderr>' mode='w' encoding='gbk'>"

stdin = None  # (!) real value is "<_io.TextIOWrapper name=3 mode='r' encoding='cp936'>"

stdout = None  # (!) forward: __stdout__, real value is "<_io.TextIOWrapper name='<stdout>' mode='w' encoding='gbk'>"

thread_info = (
    'nt',
    None,
    None,
)

version_info = (
    3,
    8,
    3,
    'final',
    0,
)

warnoptions = []

_git = (
    'CPython',
    '',
    '',
)

_xoptions = {}

__spec__ = None  # (!) real value is "ModuleSpec(name='sys', loader=<class '_frozen_importlib.BuiltinImporter'>)"

__stderr__ = stderr

__stdin__ = None  # (!) real value is "<_io.TextIOWrapper name='<stdin>' mode='r' encoding='gbk'>"

__stdout__ = None  # (!) real value is "<_io.TextIOWrapper name='<stdout>' mode='w' encoding='gbk'>"

# intermittent names
exc_value = Exception()
exc_traceback = None
