
                                        CANNONICAL SHAPE:

                            (BATCH, [CHANNEL], TIME, [DIM], [COMPONENT])


    abs(Spectro), MelSpec, MFCC, Qt, Repr:                                       (Batch, Time, Dim)

    Complex_S:                                                     (Batch, [Channel], Time, Dim, 2)

    y:                                                                     (Batch, [Channel], Time)

    enveloppe:                                                             (Batch, [Channel], Time)

    text, lyrics, file_label, segment_label:      (Batch, [Channel], Time, [Embedding, Class_Size])

    pitch, speaker_id:                            (Batch, [Channel], Time, [Embedding, Class_Size])

    qx, k_mer_hash, frame_index, cluster_label:   (Batch, [Channel], Time, [Embedding, Class_Size])

    y_bits:                                                     (Batch, [Channel], Time, Bit_Depth)



                              Modules can change SHAPES through:

        - Project:  (Batch, [Channel], Time) ----> (Batch, [Channel], Time, Dim)
        - Map/Transform:       ANY Structure ----> SAME Structure
        - Predict:                       Any ----> TRAINING: (..., Dim) BUT INFER: (..., 1)
        - Fork:            (...., Component) ----> (...) x Component
        - Join:            (...) x Component ----> (...., Component)

                             Modules can change SIGNATURES through:

        - Iso:                      N Inputs ----> N Outputs
        - Split:                    1 Input  ----> N Outputs
        - Reduce:                   N Inputs ----> 1 Outputs
        - Transform:                N Inputs ----> M Outputs


                            So, now, we can try to solve:

                        REPR Shape ---> F(...) = ??? ---> Model Shape


------

    i.e.
    repr_shape=(B=-1, T=-1)
    model_shape=(B=-1, T=-1, D=128)
                            ---> ??? = Project

    repr_shape=(B=-1, T=-1, D=1025) X N
    model_shape=(B=-1, T=-1, D=128)
                            ---> ??? = [Reduce, Map] OR [Join, Reduce], ....
                        
                        
**A. User says:**

                "I want to connect this_feat with this_network."
             
**B. Mimikit answers:**
            
                "Then you can use this_options" 
                ---> IOConfigService.resolve(this_feat, this_network_class): -> {io_module_config, ...}

**C. User chooses, configures and then clicks `Run`. mimikit goes:**

                "Let's connect all those tings" 
                --> ModelInstantiator(this_feat, this_network, this_io_config) -> Model
                



