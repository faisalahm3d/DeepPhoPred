# open(OUT,'>out.txt');
open(DATA,"/root/MPSite-18/PSSM_S/S-training-BacPhos.txt");
$tag=();$x=0;
while ($line=<DATA>){


	if ($line=~/^\>/){$x++;open (SEQ,"+>sequence.fasta");print SEQ "$line \t";}
	else {
	print SEQ "$line\n";close SEQ;
	
	`/root/blast-2.2.26/bin/blastpgp -i sequence.fasta -o result.txt -d /root/nr90_2017.02 -h 0.001 -j 3 -Q pssm$x.txt`;
		# open (FILE,"pssm$x.txt");

	}
}