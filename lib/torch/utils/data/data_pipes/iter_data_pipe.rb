module Torch
  module Utils
    module Data
      module DataPipes
        class IterDataPipe < IterableDataset
          def self.functional_datapipe(name)
            IterDataPipe.register_datapipe_as_function(name, self)
          end

          def self.functions
            @functions ||= {}
          end

          def self.register_datapipe_as_function(function_name, cls_to_register)
            if functions.include?(function_name)
              raise Error, "Unable to add DataPipe function name #{function_name} as it is already taken"
            end

            functions[function_name] = cls_to_register

            # shareable so it can be called from Ractors
            define_method function_name, &Ractor.make_shareable(proc { |*args, **options, &block|
              cls_to_register.new(self, *args, **options, &block)
            })
          end

          def reset
            # no-op, but subclasses can override
          end

          def each(&block)
            reset
            @source_datapipe.each(&block)
          end
        end
      end
    end
  end
end
