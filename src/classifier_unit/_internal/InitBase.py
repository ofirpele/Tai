class InitBase:
    
    def __init__(self, **kwargs_classifier_init):
        assert 'monotone_constraints' not in kwargs_classifier_init
        self.kwargs_classifier_init = kwargs_classifier_init