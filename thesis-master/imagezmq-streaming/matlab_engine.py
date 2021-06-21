"""
    Starts a MATLAB session, or connects to an existing MATLAB session

    To avoid the time spent starting the MATLAB engine, one can start MATLAB
    externally. But by default MATLAB starts in a non-shared mode.
    To enable sharing within MATLAB run the command:
            matlab.engine.shareEngine
"""
import matlab.engine


class MatlabSession:
    def __init__(self):
        # acquire / start the session
        self.m = False
        matlab_available = matlab.engine.find_matlab()
        if matlab_available:
            print('Found running MATLAB session(s):', matlab_available)
            self.m = matlab.engine.connect_matlab(matlab_available[0])
        else:
            print('Starting a MATLAB session')
            self.m = matlab.engine.start_matlab()       # this might be a bit slow


    def __del__(self):
        # release / close the session
        print('Releasing the MATLAB session')
        self.m.quit()
