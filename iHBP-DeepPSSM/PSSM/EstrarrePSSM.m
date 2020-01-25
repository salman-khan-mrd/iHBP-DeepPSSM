
function FEAT=EstrarrePSSM(DATA)
%DATA: cell array that store the protein
%Notice that you must download blastpgp.exe and swissprot

for i=1:size(DATA,2)
    str='C:\lavoro\TOOL\inutile.txt';
    fid=fopen(str,'w');
    fprintf(fid,'%s\n','>111');
    fprintf(fid,'%s\n',DATA{i});
    fclose(fid)
    
    %calculate PSSM
    system('C:\lavoro\Implementazioni\PSI\bin\blastpgp.exe  -i C:\lavoro\TOOL\inutile.txt  -d C:\lavoro\TOOL\CreoPSSM\PSI-BLAST\swissprot -Q C:\lavoro\TOOL\CreoPSSM\Data\PSSMut.txt -j 3');
    
    %to read PSSM from the txt file
    str='C:\lavoro\TOOL\CreoPSSM\Data\PSSMut.txt';
    fid=fopen(str,'r');
    for j=1:3
        fgetl(fid);
    end
    for j=1:length(DATA{i})
        fscanf(fid,'%f',1);
        fscanf(fid,'%c',2);
        for ij=1:20%i 20 amino-acidi
            p=fscanf(fid,'%f',1);
            if size(p,1)>0
                FEAT{i}(j,ij)=p;
            end
        end
        fgetl(fid);
    end
    fclose(fid);
    
end






















