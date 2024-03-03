function plotLearning(Loss, epochs, Acc, ax, maxIter, verbose)
        
        if size(Acc,1) == 2 && size(Loss,1) == 2 % make sure that there is both losses and accuracy scores
            train_loss = Loss(1,epochs);
            val_loss = Loss(2,epochs);
            train_acc = Acc(1,epochs);
            val_acc = Acc(2,epochs);
            
           
            % Loss plots
            plot(ax(1), epochs, train_loss, 'b');
            hold on
            plot(ax(1), epochs, val_loss, 'm');
            drawnow
            title(ax(1),'Loss History');
            xlabel(ax(1),'Epoch');
            ylabel(ax(1),'Loss')
            ax(1).XLim = [0, maxIter];
            ax(1).YLim = [0, max(Loss,[], 'all')+max(Loss,[], 'all')/100];
            %legend(ax(1), 'Training loss', 'Validation loss')
            hold off

            % Accuracy plots
            plot(ax(2), epochs, train_acc, 'b')
            hold on
            plot(ax(2), epochs, val_acc, 'm')
            title(ax(2),'Accuracy History');
            xlabel(ax(2),'Epoch');
            ylabel(ax(2),'Accuracy')
            ax(2).XLim = [0, maxIter];
            ax(2).YLim = [0,1];
            %legend(ax(2), 'Training accuracy', 'Validation accuracy')
            hold off

        else
            % Loss plot
            plot(ax(1), epochs, Loss);
            title(ax(1),'Loss History');
            xlabel(ax(1),'Epoch');
            ylabel(ax(1),'Loss')
            ax(1).XLim = [0, maxIter];
            ax(1).YLim = [0,Loss(1)+Loss(1)/100];
            legend(ax(1), 'Training loss')

            % Accuracy plot
            plot(ax(2), epochs, Acc)
            title(ax(2),'Accuracy History');
            xlabel(ax(2),'Epoch');
            ylabel(ax(2),'Accuracy')
            ax(2).XLim = [0, maxIter];
            ax(2).YLim = [0,1];
            legend(ax(2), 'Training accuracy')
        end

        if exist("verbose","var") && verbose
            disp(['Epoch #: ' num2str(epochs(end)) ' / ' num2str(maxIter)...
                ' | Training loss: ' num2str(Loss(1,end)) ' | Training Accuracy: ' num2str(Acc(1,end))]);
        end

        drawnow
end