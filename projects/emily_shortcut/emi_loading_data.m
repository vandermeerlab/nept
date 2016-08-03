% Change this filepath to where this data is located on your computer.
path = 'R066-2014-12-01_recording';

% Below works only on Emily's laptop.
% filepath = 'C:\Users\Emily\Desktop';
% savepath = 'C:\Users\Emily\Code\python-vdmlab\projects\emily_shortcut\cache\data\';

% savepath = 'C:\Users\Emily\Code\Shortcut\analysis';
% cd('C:\Users\Emily\Code\Shortcut\analysis\expkeys');
% LoadExpKeys;

% Below works on Emily's work computer
filepath = 'E:\data-shortcut\data-working\Shortcut-20150727\';
% savepath = 'E:\code\python-vdmlab\projects\emily_shortcut\cache\data\';
savepath = 'C:\Users\EmilyWork\Desktop\lfps\';


%% input_csc
cd([filepath, path(1:4), '_EI\', path]);
cfg_csc = [];
% tetrodes = {'01', '02', '03', '04', '05', '06', '07', '08', ...
%             '09', '10', '11', '12', '13', '14', '15', '16'};
tetrodes = {'01', '02', '03', '05', '06', '07', '08', ...
            '09', '10', '11', '12', '13', '15'};
for this_tt = 1:length(tetrodes)
    cfg_csc.fc = {[path(1:15), '-CSC', tetrodes{this_tt}, 'c.ncs']};
    csc = LoadCSC(cfg_csc);

    csc_type = csc.type;
    csc_tvec = csc.tvec;
    csc_data = csc.data;
    csc_label = csc.label;

    save([savepath, path(1:15), '-csc', tetrodes{this_tt}], ...
        'csc_data', 'csc_tvec', 'csc_type', 'csc_label');
end

%% input_positions
cd([filepath, '\', path]);
cfg_pos = [];
position = LoadPos(cfg_pos);

pos_type = position.type;
pos_tvec = position.tvec;
pos_datax = position.data(1,:);
pos_datay = position.data(2,:);
pos_label = position.label;

save(strcat(filepath,'\emi_inputs_position'), ...
    'pos_datax', 'pos_datay', 'pos_tvec', 'pos_type', 'pos_label')


%% input events for shortcut 

ExpKeys.led1id = 'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0001).'; 
ExpKeys.led2id = 'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0002).';
ExpKeys.ledoff = 'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0000).';
ExpKeys.pb1id = 'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0040).';
ExpKeys.pb2id = 'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0020).';
ExpKeys.pboff = 'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0000).';
ExpKeys.feeder1id = 'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0004).';
ExpKeys.feeder2id = 'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0040).';
ExpKeys.feederoff = 'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0000).';

% cd([filepath, '\', path]);
cfg_evt = [];
cfg.evt.eventList = {'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0001).'; ...
                     'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0002).'; ...
                     'TTL Output on AcqSystem1_0 board 0 port 2 value (0x0000).'; ...
                     'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0040).'; ...
                     'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0020).'; ...
                     'TTL Input on AcqSystem1_0 board 0 port 1 value (0x0000).'; ...
                     'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0004).'; ...
                     'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0040).'; ...
                     'TTL Output on AcqSystem1_0 board 0 port 0 value (0x0000).'};
cfg.eventLabel = {'led1id', 'led2id', 'ledoff', 'pb1id', 'pb2id', 'pboff', ...
                  'feeder1id', 'feeder2id', 'feederoff'};

evt = LoadEvents(cfg_evt);
evt_type = evt.type;
evt_led1id = evt.t{1};
evt_led2id = evt.t{2};
evt_ledoff = evt.t{3};
evt_pb1id = evt.t{4};
evt_pb2id = evt.t{5};
evt_pboff = evt.t{6};
evt_feeder1id = evt.t{7};
evt_feeder2id = evt.t{8};
evt_feederoff = evt.t{9};
evt_label = evt.label;

save([savepath, path(1:15), '-event'], ...
     'evt_led1id', 'evt_led2id', 'evt_ledoff', 'evt_pb1id', 'evt_pb2id', ...
     'evt_pboff', 'evt_feeder1id', 'evt_feeder2id', 'evt_feederoff', ...
     'evt_type', 'evt_label');
 

%% input_spikes
% cd([filepath, '\', path]);

cfg_spk = [];
cfg_spk.uint = '64';
cfg_spk.load_questionable_cells = 1;

spikes = LoadSpikes(cfg_spk);

spikes_type = spikes.type;
spikes_times = spikes.t;
spikes_label = spikes.label;

save([savepath, path(1:15), '-spike'], ...
    'spikes_times', 'spikes_label', 'spikes_type');
